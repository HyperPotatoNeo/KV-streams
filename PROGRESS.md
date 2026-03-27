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

#### Current runs (launched 2026-03-27 ~10:45 AM PDT)
6 conditions across 6 NERSC Perlmutter nodes (24× A100-80GB), 4-GPU DDP per condition.
All: 600 steps, lr=1e-4, WSD schedule, 50K examples/dataset, unfiltered data.

| Node | Condition | W | P | K | batch_total | Status |
|------|-----------|---|---|---|-------------|--------|
| nid008205 | **B W=512** | 512 | 64 | 8 | 512 (4×4×32) | Running |
| nid008268 | D (random kv) | 128 | 16 | 2 | 512 (4×8×16) | Running |
| nid008297 | E (no kv) | 128 | 16 | 2 | 512 (4×8×16) | Running |
| nid008304 | B K=4 | 128 | 16 | 4 | 512 (4×8×16) | Running |
| nid008448 | B P=32 | 128 | 32 | 2 | 512 (4×8×16) | Running |
| nid008480 | A (full ctx) | full | - | - | 512 (4×2×64) | Running |

Estimated: ~14-27h per blockwise condition, ~80h for A.
Checkpoints every 50 steps. 48-hour node allocations. Reservation until 2026-04-05.

| Exp | Config | Steps | val_ppl | cross_block_ppl | Notes |
|-----|--------|-------|---------|-----------------|-------|
