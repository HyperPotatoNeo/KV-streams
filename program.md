# KV Self-Compaction Phase 2: Autonomous Research Program

## Who You Are

You are an autonomous research agent implementing Phase 2 of KV Self-Compaction — scaling
the method from a 50M GPT model to Qwen3-0.6B-Base with LoRA finetuning. This document
is your complete instruction set. Read it fully before writing any code.

---

## Project Context

### What is KV Self-Compaction?

A method to turn a pretrained transformer into a blockwise RNN with constant-memory inference:

1. Process text in blocks of W tokens
2. Append P learned "compaction token" embeddings after each block
3. Forward the block (W text + P compaction tokens) through the transformer
4. Extract the KV cache entries at compaction token positions → `compact_kv`
5. Prepend `compact_kv` to the next block's attention as compressed history
6. Train end-to-end with next-token prediction loss (on text tokens only)
7. BPTT: call backward() every K blocks, then detach compact_kv

The compaction tokens learn to summarize the block's information into a fixed-size
state (P KV pairs per layer), enabling O(1) memory inference regardless of sequence length.

### Phase 1 Results (Reference: `$SCRATCH/kv-self-compaction/`)

Phase 1 validated the mechanism on a 50M-param custom GPT model. 30 experiments led to
a critical breakthrough:

**Key finding**: A learnable per-head attention bias (32 parameters, initialized to -2.0)
on compact_kv attention logits completely fixes P>1 compaction. Without it, P>1 is
*worse* than P=1 because softmax dilution overwhelms predictions with noisy compact_kv.

| Config | val_bpb | cross_block_bpb | Notes |
|--------|---------|-----------------|-------|
| Full context baseline | 0.975 | N/A | Upper bound |
| Truncation W=256 | 1.349 | 1.352 | Lower bound |
| **KV-SC K=4 P=16 ft+bias** | **0.990** | **1.054** | Best overall |
| KV-SC K=4 P=16 scratch+bias | 1.336 | 1.048 | Best cross-block |
| GatedDeltaNet (74M params) | 1.056 | 1.050 | 48% more params, 40x more state |

KV-SC matches GatedDeltaNet with **40x less state** (256KB vs 10,240KB) and **32% fewer params**.

**Reference files** (read these to understand Phase 1 patterns):
- `$SCRATCH/kv-self-compaction/train.py` — Phase 1 implementation (~1210 lines, all in one file)
  - `blockwise_train_step()` at lines ~863-967
  - `blockwise_forward_eval()` at lines ~970-1063
  - `CausalSelfAttention` with compact_attn_bias at lines ~59-125
  - Compaction embedding init at line ~216
- `$SCRATCH/kv-self-compaction/evaluate_compaction.py` — Evaluation metrics
- `$SCRATCH/kv-self-compaction/PROGRESS.md` — Full experiment log with root cause analysis

### Phase 2 Question

**Can we add compaction to a pretrained model (Qwen3-0.6B-Base) via LoRA + learned embeddings?**

Key differences from Phase 1:
- Pretrained 0.6B model (not 50M from scratch) → LoRA adaptation
- GQA (8 KV heads, 16 Q heads) → more compaction capacity per token
- 28 layers (not 8) → deeper state propagation
- Diverse SFT data (Dolci) → math, code, chat, reasoning
- HuggingFace model → must work around HF's attention mask creation
- DDP on multiple GPUs (Phase 2c) → gradient sync with BPTT

---

## Architecture: Qwen3-0.6B-Base

| Parameter | Value |
|-----------|-------|
| HF path | `Qwen/Qwen3-0.6B-Base` |
| Layers | 28 |
| Hidden size | 1024 |
| Q heads / KV heads | 16 / 8 (GQA 2:1) |
| Head dim | 128 (independent, NOT hidden_size/num_heads) |
| FFN | 3072 (SwiGLU) |
| Vocab | 151,936 |
| RoPE theta | 1,000,000 |
| BOS token | None (`add_bos_token=false`) |
| EOS/Pad token | `<\|endoftext\|>` (ID 151643) |
| Chat template | Built-in Jinja2 (`<\|im_start\|>` / `<\|im_end\|>`) |
| Think tokens | `<think>` (151667), `</think>` (151668) |

---

## Dataset: Dolci SFT Mixture

Two datasets mixed ~1:1 (OLMo3-style):

| Dataset | HF Path | Examples | Format |
|---------|---------|----------|--------|
| Think-SFT | `allenai/Dolci-Think-SFT-7B` | 2.27M | `{messages: [{role, content}]}` with `<think>` traces |
| Instruct-SFT | `allenai/Dolci-Instruct-SFT` | 2.15M | `{messages: [{role, content}]}` multi-domain |

Processing: `tokenizer.apply_chat_template(messages)` → tokenize → mask non-assistant turns
with -100 → pad to multiple of W → train on assistant tokens only.

**IMPORTANT**: Before fixing W/P/max_seq_len, run sequence length analysis (Step 1 below)
to choose values that give >= 4 blocks per median sequence.

---

## What You Can and Cannot Modify

| Path | Permission | Notes |
|------|-----------|-------|
| `src/*.py` | **EDIT** | All source code. Your workspace. |
| `tests/*.py` | **READ-ONLY by default** | See "Test Editing Protocol" below. |
| `scripts/*.py` | READ-ONLY | Pre-written launch/validation scripts. |
| `PROGRESS.md` | **EDIT** | Log every experiment result here. |
| `program.md` | READ-ONLY | These instructions. |
| `pyproject.toml` | READ-ONLY | Dependencies fixed. |

### Test Editing Protocol

Tests are immutable **by default** — they are the specification. If a test fails, fix `src/`,
not the test. However, if you believe a test has a genuine bug (wrong assertion, bad fixture,
missing import), you may edit it **only after** getting the change reviewed:

1. Identify the specific test failure and why you believe the test is wrong
2. Write your proposed fix
3. **Spawn an adversarial subagent** with:
   - The failing test code
   - Your proposed fix
   - The relevant source code the test is testing
   - Task: "Is this test wrong, or is the source code wrong? If the test is wrong, is
     the proposed fix correct? Check edge cases."
4. Only apply the fix if the adversarial agent agrees the test is wrong
5. Log the test edit in PROGRESS.md: which test, what changed, why, adversarial review result

---

## Execution Sequence

Follow these steps IN ORDER. Do not skip ahead.

### Step 0: Read Reference Code (30 min)

Before writing any code, read these files to understand patterns:

```
READ (in this order):
1. This file (program.md) — complete instructions
2. $SCRATCH/kv-self-compaction/train.py — Phase 1 implementation
   Focus on: blockwise_train_step (lines 863-967), compact_attn_bias (lines 76-77, 112-122)
3. $SCRATCH/kv-self-compaction/PROGRESS.md — Phase 1 experiment log
4. src/config.py — Phase 2 configuration
5. All other src/*.py files — understand the module structure
6. All tests/*.py files — understand what the tests expect
```

### Step 1: Validate Setup + Analyze Data (GPU, 30 min)

Run pre-flight checks to verify all assumptions hold:

```bash
# Inside container on a GPU node:
python scripts/validate_setup.py
```

This checks:
- Qwen3-0.6B-Base loads with eager attention
- Dolci datasets are accessible with expected columns
- Eager vs SDPA performance overhead
- **Sequence length analysis** → recommends W, P, max_seq_len

**ACTION**: Based on the sequence length analysis output:
- If recommended W differs from default (256), update `src/config.py` defaults
- The goal is >= 4 blocks for the median sequence length
- P should be W/8 (compression ratio from Phase 1)
- max_seq_len should cover p90 of the distribution (capped at 8192)

**Spawn an adversarial subagent** to review your W/P/max_seq_len choices:
- Give it the sequence length statistics
- Give it your proposed config changes
- Task: "Are these W/P choices sound? Will we get enough block boundaries
  for meaningful compaction training? Any edge cases with very short or very
  long sequences?"

### Step 2: Run CPU Tests (5 min)

```bash
python -m pytest tests/ -v -m "not gpu" --tb=short
```

Expected: ~30 CPU tests pass (test_config, test_kv_manager, most of test_attention, some test_data).

**If tests fail:**
- Read the error carefully
- Fix `src/` code to match what the tests expect
- Do NOT edit tests unless you follow the Test Editing Protocol above
- Re-run until all CPU tests pass

### Step 3: Run GPU Tests (15 min)

```bash
python -m pytest tests/ -v -m gpu --tb=short
```

Expected: ~58 GPU tests pass (test_model, test_blockwise, test_evaluate, test_integration,
test_gpu_specifics, GPU tests in test_attention and test_data).

**This is the critical validation step.** GPU tests verify:
- Model loads with LoRA + compaction params
- Gradient flows through compact_kv to compaction_embeddings and attn_bias
- BPTT truncation works
- Blockwise forward matches full-context for block 0
- Memory stays within budget
- Checkpoint save/load roundtrip

**If GPU tests fail:**
- Prioritize: gradient flow tests > shape tests > integration tests
- Use the debugging guide (below) to diagnose
- **Spawn an adversarial subagent** before applying any fix:
  - Give it the failing test, the error traceback, and your proposed fix
  - Task: "Will this fix break anything else? Is there a simpler fix?"

### Step 4: Smoke Test — 10 Training Steps (GPU, 10 min)

```bash
python -m src.train --condition B --max_steps 10 --eval_every 5 --batch_size 2 \
  --max_examples_per_dataset 100
```

Check:
- [ ] Loss is finite (not NaN, not inf)
- [ ] Loss decreases between step 1 and step 10
- [ ] `bias_mean` drifts from -2.0 (even slightly)
- [ ] `embed_norm` is > 0
- [ ] Evaluation runs without error at step 5 and 10
- [ ] `cross_block_ppl` is reported and finite

**If any check fails**: stop and debug before proceeding. See Debugging Guide below.

### Step 5: Phase 2a Training — Condition B (GPU, 2-4 hours)

```bash
python -m src.train --condition B --max_steps 200 --eval_every 50 \
  --max_examples_per_dataset 5000
```

**Success criteria (all must be met):**
1. Loss decreases over 200 steps
2. `compaction_embeddings.grad` is non-zero (verified by GPU tests, but double-check in logs)
3. `compact_attn_bias` has drifted from -2.0 by at least 0.01
4. `cross_block_ppl` is lower than the truncation lower bound

**After training completes:**
1. Log all results in PROGRESS.md
2. **Spawn an adversarial subagent** to review the results:
   - Give it the training logs, final metrics, and per-block PPL
   - Task: "Are these results consistent? Could the model be cheating (e.g., ignoring
     compact_kv entirely)? Is cross_block_ppl actually measuring cross-block transfer?"
3. If results look good, proceed to Step 6
4. If results are bad, see "What To Do If Compaction Doesn't Work" below

### Step 6: Add Truncation Baseline — Condition C (GPU, 1 hour)

```bash
python -m src.train --condition C --max_steps 200 --eval_every 50 \
  --max_examples_per_dataset 5000
```

This trains on only the last W tokens of each sequence. It establishes the lower bound:
if Condition B doesn't beat Condition C, compaction is not working.

### Step 7: Run All 4 Conditions — Conditions A/B/C/D (GPU, 4-8 hours)

**Can be parallelized**: Run all 4 conditions on separate GPUs simultaneously:

```bash
# GPU 0: Condition A (full context, upper bound)
CUDA_VISIBLE_DEVICES=0 python -m src.train --condition A --max_steps 200 &

# GPU 1: Condition B (KV Self-Compaction, our method)
CUDA_VISIBLE_DEVICES=1 python -m src.train --condition B --max_steps 200 &

# GPU 2: Condition C (truncation, lower bound)
CUDA_VISIBLE_DEVICES=2 python -m src.train --condition C --max_steps 200 &

# GPU 3: Condition D (random compact_kv, ablation)
CUDA_VISIBLE_DEVICES=3 python -m src.train --condition D --max_steps 200 &

wait
```

**Success criteria:**
| Criterion | Required? | Formula |
|-----------|-----------|---------|
| Compaction beats truncation | **YES** | cross_block_ppl(B) < cross_block_ppl(C) |
| Model uses state | **YES** | cross_block_ppl(D) > cross_block_ppl(B) by >5% |
| State carries across blocks | YES | per_block_ppl ratio (max/min) < 2.0x |
| Quality preserved | DESIRED | val_ppl(B) < val_ppl(A) * 1.05 |

**After all conditions complete:**
1. Log complete comparison table in PROGRESS.md
2. **Spawn an adversarial subagent** to review the comparison:
   - Give it all 4 conditions' metrics
   - Task: "Are these comparisons fair? Same data, same steps, same hyperparameters?
     Could any condition have an unfair advantage? Are the metrics computed correctly
     (e.g., does Condition A's cross_block_ppl make sense)?"

### Step 8: Scale-Up — More Data, More Steps (GPU, 4-8 hours)

If Phase 2a succeeds, scale up:
```bash
python -m src.train --condition B --max_steps 660 --max_examples_per_dataset 50000 \
  --grad_accum 4
```

Also run Conditions A and C at the same scale for comparison.

---

## Parallelization Guide

### What CAN be parallelized (separate GPUs):
- Multiple conditions (A/B/C/D) running simultaneously on different GPUs
- Multiple hyperparameter sweeps (different W, P, K values)
- Training + evaluation (eval is fast, runs inline)

### What CANNOT be parallelized:
- Steps 0-4 (must be sequential — each depends on the previous)
- Tests → smoke test → training (must validate before scaling)
- Reading code → writing code (understand before implementing)

### How to parallelize on NERSC:
```bash
# Allocate 1 node with 4 GPUs
salloc --reservation=_CAP_tinker -A m5017 -C "gpu&hbm80g" --gpus-per-node=4 --nodes=1 --time=4:00:00

# Inside container, launch 4 conditions on 4 GPUs:
for i in 0 1 2 3; do
  CONDITION=$(echo "A B C D" | cut -d' ' -f$((i+1)))
  CUDA_VISIBLE_DEVICES=$i python -m src.train --condition $CONDITION \
    --output_dir outputs/$CONDITION &
done
wait
```

---

## Testing Strategy

### Test Hierarchy (run in this order)

1. **CPU unit tests** (`-m "not gpu"`): Fast, catch shape/logic errors. Run after ANY code change.
2. **GPU unit tests** (`-m gpu`): Catch model-specific errors. Run after model/blockwise changes.
3. **Smoke test** (10 steps): Catch training loop errors. Run before any long training.
4. **Short training** (50-200 steps): Catch convergence issues. The main validation.

### When to Run Which Tests

| Action | Run |
|--------|-----|
| Changed config.py | CPU tests only |
| Changed data.py | CPU tests + `test_data.py` GPU tests |
| Changed model.py | All GPU tests |
| Changed attention.py | `test_attention.py` + `test_blockwise.py` (GPU) |
| Changed kv_manager.py | `test_kv_manager.py` (CPU) + `test_blockwise.py` (GPU) |
| Changed blockwise.py | `test_blockwise.py` + `test_integration.py` (GPU) |
| Changed train.py | Smoke test (10 steps) |
| Changed evaluate.py | `test_evaluate.py` (GPU) |
| Before any training run | ALL tests |

### Test Failure Triage

| Test category | If failing, likely bug in: |
|---------------|--------------------------|
| test_config | config.py — trivial to fix |
| test_attention mask shape | attention.py build_4d_attention_mask |
| test_attention causal | attention.py mask construction logic |
| test_kv_manager extract | kv_manager.py — check DynamicCache API |
| test_model loads | model.py — check peft wrapper navigation |
| test_blockwise gradient | blockwise.py — check .clone(), BPTT logic |
| test_integration end-to-end | train.py — check condition routing, optimizer |
| test_gpu memory | Likely batch_size too large, or BPTT not detaching |

---

## Debugging Guide

### Loss NaN at Step 1
**Cause**: Forward pass or mask bug.
**Debug**:
1. Print attention mask shape: should be `(B, 16, W+P, P_past+W+P)`
2. Print position_ids shape: MUST be 2D `(B, seq_len)`, not 1D
3. Print logits: check for inf values before cross_entropy
4. Try block 0 only (no compact_kv) — if block 0 NaN, the mask or forward is broken
5. **Spawn adversarial subagent**: give it the shapes and mask values, ask "is this mask valid?"

### Loss Not Decreasing
**Cause**: Model ignoring compact_kv, or gradient not reaching compaction params.
**Debug**:
1. Check `model.compaction_embeddings.grad` — if None, backward didn't reach it
2. Check `model.compact_attn_bias.grad` — if None, mask isn't using the bias
3. Run Condition D and compare to B — if D ≈ B, model is ignoring compact_kv
4. Print attention weights at compact_kv positions — if all near zero, bias too negative
5. Try setting `bias_init = 0.0` temporarily to see if that changes behavior

### cross_block_ppl Worse Than Truncation
**Cause**: Compaction is actively hurting, not helping.
**Debug**:
1. Check per_block_ppl breakdown — is block 1 much worse than block 0?
2. Check if compact_kv norms are reasonable (not exploding or vanishing)
3. Phase 1 root cause was softmax dilution — verify bias is being applied
4. Print the 4D mask for block 1: columns 0:P should have bias ≈ -2.0, not -inf

### OOM After N Steps
**Cause**: Gradient graph not being truncated by BPTT.
**Debug**:
1. Check that `detach_compact_kv()` is called after every K blocks
2. Check that `.clone()` is used in `extract_compact_kv` (not raw slices)
3. Monitor `torch.cuda.max_memory_allocated()` — should be constant, not growing
4. Try K=1 (detach after every block) to isolate

### Import Errors
**Fix**:
```bash
uv pip install peft datasets   # if peft or datasets missing
```

---

## Adversarial Subagent Protocol

**Use adversarial subagents frequently.** They catch bugs you can't see because you're
too close to the code. The cost is ~3 minutes per review; the cost of a silent bug is
hours of wasted GPU time.

### When to Spawn Adversarial Subagents

| Trigger | What to review |
|---------|---------------|
| Before changing W/P/max_seq_len | "Are these choices sound for the data distribution?" |
| Before any `src/` code change > 10 lines | "Will this break anything? Is there a simpler approach?" |
| Before editing a test file | "Is the test wrong, or is the source wrong?" |
| After a test failure you can't immediately diagnose | "Here's the error — what's the root cause?" |
| After each training experiment | "Are these results consistent? Could the model be cheating?" |
| Before committing to a debugging hypothesis | "Is this the right diagnosis? What else could cause this?" |
| When changing attention mask logic | "Is this mask correct for all block indices? Edge cases?" |
| When changing BPTT/gradient flow | "Does gradient flow correctly through compact_kv?" |

### How to Spawn an Adversarial Subagent

Give the subagent:
1. **The specific code** it's reviewing (not the whole file — just the relevant function)
2. **The context** — what this code does, what changed, why
3. **The actual source code it depends on** — file paths so it can read them
4. **A clear task** — "Find problems. Things that will break. Things that are wrong."

Do NOT:
- Give the subagent your reasoning for why the code is correct (that's confirmation bias)
- Ask "is this OK?" (too easy to say yes) — ask "what's wrong with this?"
- Skip the review because "it's a small change" (small changes cause the worst bugs)

### Adversarial Subagent for Test Edits

This is the ONLY path to editing test files:

```
1. I found test_X failing with error Y
2. I believe the test is wrong because Z
3. My proposed fix: [exact diff]
4. SPAWNING ADVERSARIAL SUBAGENT:
   - Here is the failing test code
   - Here is the source code it tests
   - Here is my proposed test fix
   - Task: "Is the test wrong, or is the source wrong? If the test is wrong,
     is this fix correct? Does it still test what it was supposed to test?"
5. Subagent says: [result]
6. If approved: apply fix, log in PROGRESS.md
   If rejected: fix src/ instead
```

---

## Invariants (NEVER Violate These)

These are hard-won lessons from Phase 1. Each one caused hours of wasted GPU time when
violated. Read each one and understand WHY.

### 1. gradient_checkpointing is FORBIDDEN
**Why**: Calling `model.gradient_checkpointing_enable()` silently sets `use_cache=False`
in the model config. This means the DynamicCache is never populated, compact_kv is always
empty, and compaction silently stops working. The model trains normally on individual blocks
but cross-block transfer is zero. This is undetectable from loss curves alone.

### 2. attn_implementation MUST be "eager"
**Why**: SDPA (Scaled Dot Product Attention) with `is_causal=True` uses a fast kernel that
ignores any explicit attention mask. Our 4D mask with learnable bias on compact_kv columns
gets silently dropped. Eager attention explicitly computes `attn_weights + causal_mask`,
which is the only reliable way to inject the per-head bias.

### 3. position_ids MUST be 2D (batch_size, seq_len)
**Why**: `Qwen3RotaryEmbedding.forward()` does `position_ids[:, None, :]` which requires
exactly 2 dimensions. 1D position_ids will be interpreted as (seq_len,) and the `[:, None, :]`
will produce wrong shapes, causing a silent computation error (not a crash).

### 4. compact_kv extraction MUST use .clone()
**Why**: `DynamicCache` stores KV internally as `self.keys = torch.cat([self.keys, new_k])`.
When you extract a slice `cache.keys[:, :, -P:, :]`, it's a *view* into the internal tensor.
On the next block, `torch.cat()` overwrites `self.keys` with a new tensor, invalidating the
view. `.clone()` creates an independent copy that preserves the gradient graph (CloneBackward).

### 5. DynamicCache() MUST be called without config=
**Why**: Passing `config=` to DynamicCache causes it to inspect `config.layer_types` and
potentially create `SlidingWindowLayer` instances instead of `DynamicLayer`. These have
different eviction behavior that breaks our extraction logic.

### 6. Bypass Qwen3Model.forward() — call decoder layers directly
**Why**: `Qwen3Model.forward()` internally calls `create_causal_mask()` which creates its
own 4D mask, potentially overriding ours. By calling `model.layers[i](...)` directly, we
have full control over the attention mask each layer receives.

### 7. Labels are NOT shifted — y already has the correct targets
**Why**: Unlike Phase 1 which used a pre-shifting dataloader, Phase 2's `data.py` creates
labels that are already aligned: `labels[i]` is the target for `input_ids[i]`. Do NOT do
`labels = input_ids[:, 1:]` — that double-shifts and silently misaligns everything.

### 8. Loss uses reduction='sum', normalized by window_tokens
**Why**: With BPTT, we accumulate loss across K blocks before calling backward(). Using
`reduction='mean'` would give incorrect per-token loss when blocks have different numbers
of valid tokens (due to -100 masking). `reduction='sum'` + explicit normalization is correct.

---

## What To Do If Compaction Doesn't Work

If Condition B fails to beat Condition C (truncation), don't panic. Phase 1 took 30 experiments
to find the right config. Try these IN ORDER:

1. **Verify bias is applied**: Print the 4D mask at block 1. Columns 0:P should show the
   bias value (≈ -2.0), not -inf or 0.0.

2. **Verify gradient flow**: Check `compaction_embeddings.grad` after one training step.
   If None or all zeros, the backward pass isn't reaching compaction params.

3. **Try different bias_init**: Phase 1 found -2.0 optimal, but this may differ for 0.6B.
   Try -1.0, -3.0, -5.0.

4. **Try different W**: If median sequence length is short, W=256 gives too few blocks.
   Try W=128 or W=64.

5. **Try different P**: Phase 1 showed monotonically better with more P (when bias works).
   But start with P = W/8.

6. **Try different K**: K=2 is conservative. K=4 allows longer gradient flow. K=1 tests
   whether the model can use state even without learning to produce it.

7. **Check data**: Are most assistant responses shorter than W? If so, there's nothing to
   compress across blocks. Filter for examples with >= 2*W tokens.

8. **Check learning rates**: Compaction embeddings and bias have separate LR groups.
   If lr_compaction is too low, embeddings learn too slowly. If lr_bias is too high,
   bias oscillates.

For each experiment, **spawn an adversarial subagent** to review your hypothesis before
running the experiment. The subagent should check: "Is this the right thing to try next,
or is there a simpler explanation for the failure?"

---

## Compute Environment

**NERSC Perlmutter**, A100-80GB GPUs.

```bash
# Allocate GPU node (reservation available until 2026-03-29)
salloc --reservation=_CAP_tinker -A m5017 -C "gpu&hbm80g" \
  --gpus-per-node=4 --nodes=1 --time=4:00:00

# Enter container
export HOME=$SCRATCH
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
cd $SCRATCH

podman-hpc run --rm -it \
  --user "$(id -u):$(id -g)" --replace --name kv-phase2 \
  --group-add keep-groups --userns keep-id --gpu --nccl --shm-size=8g \
  -e SCRATCH -e HOME -e CUDA_VISIBLE_DEVICES=0 \
  -e UV_CACHE_DIR=$SCRATCH/uv-cache \
  -v "$SCRATCH":"$SCRATCH" -v /global/homes/s/siddart2:/global/homes/s/siddart2 \
  -w "$SCRATCH/kv-self-compaction-phase2" \
  docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 \
  bash -c 'unset NCCL_SOCKET_IFNAME && bash'

# Inside container — install deps once:
uv pip install peft datasets
```

- Account: `-A m5017` for all SLURM jobs
- Reservation: `_CAP_tinker` (faster allocation, until 2026-03-29)
- Package manager: `uv` (not pip/conda) inside container
- Use `CUDA_VISIBLE_DEVICES=N` to select specific GPU (0-3)
- 4 GPUs per node → 4 parallel experiments

---

## Progress Tracking (CRITICAL — for crash recovery)

After EVERY experiment or significant code change:

1. Update `PROGRESS.md` with:
   - What you did (config, command, duration)
   - Results (all metrics)
   - Interpretation (what does this mean?)
   - Next step (what will you try next and why?)

2. If you crash/restart:
   - Read `program.md` first (this file)
   - Read `PROGRESS.md` to see what was already done
   - Read test results to see current code health
   - Resume from where you left off

---

## Rules

1. ALWAYS run tests before training. ALL tests must pass.
2. ALWAYS log results in PROGRESS.md after each experiment.
3. ALWAYS use adversarial subagents before:
   - Code changes > 10 lines
   - Test edits (mandatory)
   - Interpreting surprising results
   - Choosing next experiment after a failure
4. NEVER enable gradient_checkpointing.
5. NEVER use attn_implementation other than "eager".
6. NEVER pass config= to DynamicCache().
7. NEVER use 1D position_ids (MUST be 2D).
8. NEVER use raw cache slices for compact_kv (MUST .clone()).
9. NEVER edit tests without adversarial review + PROGRESS.md log.
10. Keep changes small and incremental. One bug fix at a time.
11. If stuck for > 30 minutes on a bug, spawn an adversarial subagent to help diagnose.
12. When in doubt, compare behavior to Phase 1's `train.py` — it's the known-working reference.
