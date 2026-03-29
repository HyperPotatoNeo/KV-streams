# KV Self-Compaction Inference: Progress Log

## Current Status: All Evaluations Complete

**Date**: 2026-03-28

---

## Final Results

### SFT Training Perplexity (Step 600)

| Method | val_ppl | cross_block_ppl | full_ctx_ppl |
|--------|---------|-----------------|--------------|
| Full Context SFT | 44.18* | 1349.0* | **3.26** |
| Learned Compaction (K=8) | **3.65** | **4.59** | N/A |
| Learned Compaction (K=1) | 4.14 | 9.70 | N/A |
| No Cross-Block (E) | 4.16 | 12.96 | N/A |

*A's val/cross_block ppl misleadingly high (blockwise eval); true perf is full_ctx_ppl=3.26*

### GSM8K Accuracy (Step 600, 50 examples, 4096 max tokens)

| Method | Correct | Accuracy | vs Full |
|--------|---------|----------|---------|
| Full Context SFT | 16/50 | **32.0%** | 100% |
| Learned Compaction (K=8) | 14/50 | **28.0%** | 87.5% |
| No Cross-Block (E) | 9/50 | **18.0%** | 56.3% |
| Attn Matching (post-hoc) | 5/50 | **10.0%** | 31.3% |
| Learned Compaction (K=1) | 4/50 | **8.0%** | 25.0% |

Full report: `report/phase2_results.md`

---

## Implementation Summary

### Inference Engine (7 modules)
- `src/inference/engine.py` — CompactionInferenceEngine (blockwise generation)
- `src/inference/masks.py` — Decode + compaction masks
- `src/inference/compat.py` — Transformers v5 compatibility
- `src/inference/server.py` — FastAPI OpenAI-compatible server
- `src/inference/sampling.py` — Temperature/top-p/logprob
- `src/inference/convert_checkpoint.py` — Training → inference format
- `src/inference/config.py` — Configuration

### Testing
- **45/45 tests pass** (19 mask + 11 server API + 15 GPU engine)
- **4 adversarial review rounds**: 7 bugs found and fixed, round 4 clean
- **Logit equivalence verified** vs training code's blockwise_forward_eval

### Eval Scripts
- `eval/gsm8k_generic.py` — Generic compaction model eval (multi-GPU)
- `eval/gsm8k_condition_a.py` — Full context model eval
- `eval/gsm8k_condition_e.py` — No cross-block eval
- `eval/gsm8k_attn_matching.py` — Attention matching compression eval
- `eval/quick_gsm8k_parallel.py` — Quick parallel eval

---

## Timeline

### 2026-03-28: Full Day

1. **Planning** (3h): Codebase exploration (3 agents), algorithm design, 2 adversarial + 1 testing specialist reviews
2. **Implementation** (2h): 7 modules, 3 test files, v5 compat fixes
3. **Adversarial bug fixing** (1h): 4 rounds, 7 bugs → 0
4. **GPU testing** (1h): 45/45 tests, logit equivalence, checkpoint conversion
5. **GSM8K evals** (~8h): All 5 conditions on interactive nodes, 4 GPUs each

### Cleanup
- Deleted old runs: `phase2a_v2/`, `smoke_test/`
- Cleared: wandb/, .pytest_cache/, logs/
- Kept: 3 checkpoint dirs (ddp_scaleup_W512/{A,B}, ddp_scaleup_W512_K1/B, ddp_scaleup_W512_E/E)

---

## SFT Training Checkpoints (Kept)

| Directory | Condition | W | P | K | Steps |
|-----------|-----------|---|---|---|-------|
| `outputs/ddp_scaleup_W512/condition_A/` | Full Context SFT | 512 | 64 | N/A | 600 |
| `outputs/ddp_scaleup_W512/condition_B/` | Learned Compaction | 512 | 64 | 8 | 600 |
| `outputs/ddp_scaleup_W512_K1/condition_B/` | Learned Compaction (shallow) | 512 | 64 | 1 | 600 |
| `outputs/ddp_scaleup_W512_E/condition_E/` | No Cross-Block | 512 | 64 | 1 | 600 |
