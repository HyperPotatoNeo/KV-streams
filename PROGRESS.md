# KV Self-Compaction Phase 2: Experiment Progress

## Overview

Scaling KV Self-Compaction from 50M GPT (Phase 1) to Qwen3-0.6B-Base with LoRA + learned
compaction embeddings. SFT on Dolci Think-SFT + Instruct-SFT mixture.

## Phase 1 Reference (50M GPT, 30 experiments)

Best config: K=4 P=16 finetune + bias → val_bpb=0.990 (baseline 0.975), cross_block_bpb=1.054
KV-SC matches GatedDeltaNet with 40x less state (256KB vs 10,240KB).

## Phase 2 Experiments

### Setup Validation (2026-03-26)

**Sequence Length Analysis** (2000 samples per dataset):
- Think-SFT: min=49, median=1812, mean=2921, max=64045
- Instruct-SFT: min=24, median=432, mean=595, max=5863
- Combined: p10=161, p25=395, **p50=997**, p75=1926, **p90=3886**, p95=5952, p99=13129

**Config changes from defaults** (adversarial reviewed):
- W: 256 → **128** (7 blocks at median vs 3)
- P: 32 → **16** (W/8 compression ratio; 28-layer GQA has 14x state/token vs Phase 1)
- max_seq_len: 2048 → **4096** (covers p90)
- Added min_blocks=3 filter in load_data (skip sequences < 384 tokens)

**Eager/SDPA overhead**: 1.34x (acceptable)

**Test edits** (both adversarial reviewed):
1. test_config.py: Updated default assertions for W=128, P=16, max_seq_len=4096. Fixed test_config_override to use W=64, P=8 (avoid degenerate overlap with new defaults).
2. test_gpu_specifics.py: Fixed DynamicCache API — `key_cache`/`value_cache` removed in transformers 4.57.6, replaced with `cache[layer_idx]` public API.

**Source edits**:
1. config.py: Updated W, P, max_seq_len defaults
2. data.py: Added min_blocks filter in load_data (not _process_example)
3. model.py: torch_dtype→dtype (deprecation fix)

**Test results**: 39/39 CPU ✅, 50/50 GPU ✅

**Source fix**: data.py: sanitize messages (skip None content, strip function_calls/functions keys from Dolci-Instruct-SFT)

### Experiments

| Exp | Config | Steps | val_ppl | cross_block_ppl | Notes |
|-----|--------|-------|---------|-----------------|-------|
| 0 | B smoke W=128 P=16 K=2 bs=2 100ex | 10 | 4.36 | 3.87 | **BUGGY** — labels[i]=token_ids[i] (identity, no shift). Model predicts current token. |

### BUG FIX: Label Alignment (2026-03-27)

**Root cause**: `data.py` line 176 set `labels[j] = token_ids[j]` (identity). Blockwise loss computes
`CE(logits[i], labels[i])` without shifting. Model trivially predicts current token → ppl ≈ 1.0.

**Evidence**: B ≈ D (both ppl ~1.01). Verified: `(labels[valid] == input_ids[valid]).mean() = 1.0`.

**Fix**: `labels[j] = token_ids[j+1]` (pre-shifted for next-token prediction). Updated test_data.py
to verify `labels[i] == input_ids[i+1]`. All 89 tests pass after fix.

**Test edit** (adversarial reviewed): test_data.py::test_label_masking_assistant_turns assertion
changed from `labels[i] == input_ids[i]` to `labels[i] == input_ids[i+1]`.

### Experiments (v2 — with correct labels)

| Exp | Config | Steps | val_ppl | cross_block_ppl | Notes |
|-----|--------|-------|---------|-----------------|-------|
| 1 | B smoke v2 W=128 P=16 K=2 bs=2 100ex | 10 | 15.2 | 34.7 | Realistic loss=2.1. Per-block ppl increases 3→18. |
| 2 | **B** (learned) W=128 P=16 K=2 bs=4 5Kex | 200 | **6.00** | **8.44** | bias=-0.93. Blocks 1-31 ppl ~5.5-6.4 (uniform). |
| 3 | C (truncation) W=128 bs=4 5Kex | 150 | ~245 | ~1405 | Only 128 tokens context. No compaction. |
| 4 | **D** (random kv) W=128 P=16 K=2 bs=4 5Kex | 200 | **7.27** | **15.26** | bias=-2.13 (suppressed). B beats D by 45%. |
| 5 | A (full ctx) W=128 bs=1 ga=4 5Kex | 200 | 508 | 3444 | Never trained with blockwise. Blockwise eval incompatible. |

### Phase 2a Results Summary (2026-03-27)

**KV Self-Compaction works at 0.6B scale!**

| Criterion | Formula | Result | Required? |
|-----------|---------|--------|-----------|
| Beats truncation | cb_ppl(B) < cb_ppl(C) | 8.44 < 1405 (167x) | **YES ✅** |
| Uses learned state | cb_ppl(D) > cb_ppl(B) by >5% | 15.26 vs 8.44 (45%) | **YES ✅** |
| State carries | max/min per_block_ppl < 2.0x | 6.42/5.51 = 1.17x | **YES ✅** |
| Quality preserved | val_ppl(B) < val_ppl(A)*1.05 | 6.0 < 508*1.05 (trivially) | **YES ✅** (unfair: A not blockwise-trained) |

**Key mechanism**: bias_mean B=-0.93 (attending to compact_kv) vs D=-2.13 (suppressing random noise).
Per-block ppl is remarkably uniform (5.5-6.4) — compaction carries information across all blocks.

**Adversarial review findings** (2026-03-27):
1. A/C comparison unfair (not trained blockwise). B vs D is the fairest comparison.
2. Need Condition E (blockwise, no compact_kv) as proper baseline.
3. The 45% B-vs-D gap + bias divergence is genuine evidence of working compaction.
4. 200 steps on 10% data is marginal — scale-up needed.

### Scale-Up (Phase 2b, 2026-03-27)

#### Infrastructure changes
- **DDP support**: Manual gradient all-reduce (no DDP wrapper — avoids .module indirection
  and multiple-backward-per-step issues with BPTT). Adversarial reviewed.
- **Condition E**: Blockwise with no compact_kv (within-block only, fairest baseline for B)
- **WSD LR schedule**: 50-step warmup → stable → 20% cosine decay (replaced cosine-with-warmup)
- **Per-step logging**: W&B + train_loss.jsonl every step (was every 10)

#### Bug fixes (all adversarial reviewed, 89/89 tests pass)
1. **A/C double-shift labels** (BLOCKING): HF's model(labels=...) internally shifts, but our
   labels are pre-shifted → model predicted 2 tokens ahead. Fixed: manual CE loss for A/C.
2. **Padding not masked in attention** (MODERATE): Padding tokens participated in attention.
   Fixed: build_4d_attention_mask now accepts padding_mask, sets padding KV columns to -inf.
3. **collate_fn pad token** (MODERATE): Padded with token ID 0 (real vocab token) instead of
   151643 (Qwen3 EOS/pad). Fixed: pass tokenizer.pad_token_id to collate_fn.
4. **Embed norm clamp timing**: Was before optimizer.step() (clamped values overwritten).
   Moved to after.
5. **set_epoch**: Started at 1, skipping epoch 0. Fixed.
6. **all_reduce_gradients**: Iterated frozen params. Added requires_grad filter.
7. **Min-length filter removed**: Was filtering sequences < 3*W tokens. Removed to allow
   all sequences (short ones still train LoRA, only multi-block sequences train compaction).
8. **Integration test updated**: test_integration.py A/C conditions now use manual CE loss
   matching train.py production code.

#### Configuration iterations
- **v1** (10:45 AM): 6 conditions (B W=512, D/E/K4/P32 W=128, A). B OOM'd at batch_size=4.
- **v2** (11:55 AM): All W=512 (B/D/E/C/A). All OOM'd at batch_size=4 for W=512 blockwise.
  A OOM'd at batch_size=2 for full 4096-token attention.
- **v3** (12:04 PM): batch_size=2 for all W=512 blockwise. A batch_size=1.
  B works. A crashed: port conflict (29501 in use), then stale GPU memory.
- **v4** (CURRENT, 1:05 PM): Focus on B + A only. 3 nodes each (12 GPUs per run).
  Fresh node allocation for A. Port 29502. Added full_context_ppl to A's eval.

#### Current runs (launched 2026-03-27 ~1:05 PM PDT)
2 runs across 6 NERSC Perlmutter nodes (24× A100-80GB), 12-GPU DDP per run.
Both: 600 steps, lr=1e-4, WSD schedule (50-step warmup), 50K examples/dataset, unfiltered data.

| Nodes | Condition | Config | batch_total | Status |
|-------|-----------|--------|-------------|--------|
| nid008304+205+268 | **B** (compaction) | W=512 P=64 K=8, bs=2, ga=22 | 528 | **Step ~100, loss=1.38, bias=-1.21** |
| nid008297+448+480 | **A** (full ctx) | Full 4096 tokens, bs=1, ga=43 | 516 | **Step ~2, training started** |

B timing: ~28s/step → 600 steps ≈ 4.7 hours (+ eval pauses).
A timing: TBD (batch_size=1 with full context is slow).
Eval at steps 100, 200, 300, 400, 500, 600. Checkpoints every 50 steps.

Other conditions (C/D/E) deferred — will run on freed nodes after B/A complete.

#### Additional code changes (v4)
- evaluate.py: Added `full_context_ppl` metric for condition A (standard HF forward, no blockwise)
- run_A_3node.sh: port 29502→29504, PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (fixes OOM)
- All W=512 blockwise conditions require batch_size=2 (batch_size=4 OOMs with K=8 eager attention)
- Dataset caching: tokenized data saved to data_cache/*.pt (~4-5 GB), loads in ~2s vs ~35 min

#### Results so far (runs still in progress)

| Exp | Metric | Step 100 | Step 200 | Step 300 | Step 400 |
|-----|--------|----------|----------|----------|----------|
| B val_ppl | blockwise | 4.224 | 3.914 | 3.795 | 3.727 |
| B cb_ppl | cross-block | 7.06 | 5.44 | 5.01 | 4.80 |
| B bias | attn bias | -1.211 | -1.078 | -1.055 | -1.047 |
| A full_ctx_ppl | standard fwd | 3.484 | 3.384 | — | — |

**Key findings** (all experiments complete, 600 steps each):
1. B val_ppl (3.652) is **12% from A's full_ctx_ppl (3.260)** — compaction approaches full-context
2. K=1 ≈ E (4.139 vs 4.158) — **BPTT is critical**, without it compact_kv doesn't help
3. B cross_block_ppl (4.59) vs E (12.96) — **2.8x improvement** from learned compaction
4. Report generated: report/report.pdf (4 pages with plots)

#### Upcoming experiments (after B+A finish)

1. **B K=1** (3 nodes): compact_kv flows but gradient detached every block. Tests whether
   explicit temporal gradient is needed, or if shared-parameter learning suffices.
   - Adversarial note: compaction_embeddings still learn indirectly via shared params.

2. **E isolation** (3 nodes, K=1): each block in isolation, no compact_kv between blocks.
   Compaction tokens still present as within-block auxiliary capacity but KV never reused.
   - Adversarial note: NOT equivalent to P=0 — compaction tokens add within-block capacity.
   - K=1 used instead of K=8 (no cross-block dependency, saves memory).

#### Scale-up experiment results

| Condition | val_ppl | cross_block_ppl | full_ctx_ppl | bias | Steps | Notes |
|-----------|---------|-----------------|-------------|------|-------|-------|
| **B** (K=8, W=512) | **3.652** | **4.59** | — | -1.016 | 600 | Full BPTT compaction |
| **K=1** (no temporal grad) | 4.139 | 9.70 | — | -1.242 | 600 | compact_kv flows but no gradient through time |
| **E** (isolation) | **4.158** | **12.96** | — | -2.000 | 600 | Each block independent, no compact_kv |
| **A** (full context) | 44.18* | — | **3.260** | -2.0 | 600 | *blockwise eval (meaningless for A) |

**Key findings:**
1. B val_ppl (3.652) is **12% from A's full_ctx_ppl (3.260)** — compaction approaches full-context quality
2. K=1 val_ppl (4.139) is **13% worse than B** — temporal gradient (BPTT) significantly helps
3. K=1 cross_block_ppl (9.70) is **2.1x worse than B (4.59)** — BPTT crucial for cross-block transfer
4. K=1 bias (-1.24) converged higher than B (-1.02) — less trust in compact_kv without temporal training

#### B eval trajectory (W=512 P=64 K=8)

| Step | val_ppl | cb_ppl | bias |
|------|---------|--------|------|
| 100 | 4.224 | 7.06 | -1.211 |
| 200 | 3.914 | 5.44 | -1.078 |
| 300 | 3.795 | 5.01 | -1.055 |
| 400 | 3.727 | 4.80 | -1.047 |
| 500 | 3.682 | 4.67 | -1.023 |
| 600 | 3.652 | 4.59 | -1.016 |

#### K=1 eval trajectory (W=512 P=64 K=1)

| Step | val_ppl | cb_ppl | bias |
|------|---------|--------|------|
| 100 | 4.919 | 15.99 | -1.500 |
| 200 | 4.433 | 12.09 | -1.367 |
| 300 | 4.298 | 10.89 | -1.297 |
| 400 | 4.224 | 10.36 | -1.273 |
| 500 | 4.171 | 9.95 | -1.266 |
| 600 | 4.139 | 9.70 | -1.242 |

#### A eval trajectory (full context)

| Step | full_ctx_ppl |
|------|-------------|
| 100 | 3.484 |
| 200 | 3.384 |
| 300 | 3.336 |
| 600 | 3.260 |

---

## Phase 2c: GSM8K Downstream Evaluation (2026-03-28)

### Inference Engine
Built custom HuggingFace-based inference engine (transformers v5.4.0, eager attention).
7 modules, 45/45 tests, 4 adversarial review rounds (7 bugs found and fixed).
Logit equivalence verified vs training code. See `src/inference/` and `eval/progress.md`.

### GSM8K Results (Step 600, 50 examples, 4096 max tokens, greedy, 4-GPU parallel)

| Method | val_ppl | GSM8K | vs Full |
|--------|---------|-------|---------|
| Full Context SFT | 3.26 (full_ctx) | **32.0%** | 100% |
| Learned Compaction K=8 | 3.65 | **28.0%** | 87.5% |
| No Cross-Block (E) | 4.16 | **18.0%** | 56.3% |
| Attn Matching (post-hoc) | N/A | **10.0%** | 31.3% |
| Learned Compaction K=1 | 4.14 | **8.0%** | 25.0% |

**Key finding**: Learned compaction (K=8) retains 87.5% of full-context GSM8K accuracy at 8x KV compression.
BPTT depth critical: K=8 (28%) >> K=1 (8%). Learned >> post-hoc attention matching (28% vs 10%).
Full report: `report/phase2_results.md`
