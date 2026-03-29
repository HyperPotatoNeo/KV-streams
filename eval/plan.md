# KV Self-Compaction Inference Plan

**Last updated**: 2026-03-28
**Status**: ALL COMPLETE — inference engine built, 5 conditions evaluated on GSM8K

## Overview

Build inference engine for KV Self-Compaction trained models. Serves OpenAI-compatible API for prime-rl RL training and verifiers gsm8k evaluation.

## Architecture

- **Framework**: HuggingFace Transformers (latest v5.4.0) with eager attention + torch.compile
- **Generation**: Custom blockwise generation loop reusing training code
- **Server**: FastAPI with prime-rl-compatible endpoints
- **Venv**: Separate project venv (not prime-rl's)

## Algorithm

After W=512 text tokens, P=64 learned compaction embeddings are forwarded through the model. The resulting compact_kv (P KV entries/layer) replaces the full cache, bounding memory.

### Three Forward Pass Types

1. **Full block prompt** (W+P queries): Uses `build_4d_attention_mask()` from training. Lower-tri causal + bias on compact_kv.
2. **Single-token decode** (1 query): `build_decode_mask()` — bias on compact_kv columns, 0.0 elsewhere.
3. **Compaction-only** (P queries): `build_compaction_mask()` — bias on compact_kv, ALL W text visible to ALL P queries, causal among compaction tokens.

### Key Rules
- Partial blocks (< W tokens) processed WITHOUT compaction (model never trained on partial blocks)
- `logprob[x_n] = log_softmax(logits_{n-1})[x_n]` (NOT step n)
- Position IDs: `block_idx * (W + P) + offset`
- GPU access serialized with asyncio.Lock

## Code Structure

```
kv-self-compaction-phase2/
  eval/
    plan.md              # This file
    progress.md          # Current status
  src/
    inference/
      __init__.py
      engine.py          # CompactionInferenceEngine
      masks.py           # build_decode_mask(), build_compaction_mask()
      server.py          # FastAPI OpenAI-compatible server
      config.py          # InferenceConfig
      sampling.py        # Top-k, top-p, temperature, logprob
      convert_checkpoint.py  # Training ckpt → PEFT + compaction params
      compat.py          # Transformers v5 compatibility layer
  tests/
    test_inference/
      test_masks.py
      test_engine.py
      test_server.py
      test_convert.py
      test_compat.py
      test_reward_hacks.py
```

## Server Endpoints (prime-rl compatible)

```
POST /v1/chat/completions          # Standard + extra_body.return_token_ids
POST /v1/chat/completions/tokens   # Token-aware (takes tokens: list[int])
POST /update_weights               # Body: {"weight_dir": "..."} → {"status": "ok"}
POST /load_lora_adapter            # Body: {"lora_name": "...", "lora_path": "..."}
POST /tokenize                     # Body: {model, messages} → {count, tokens}
GET  /health                       # → {"status": "ok"}
GET  /v1/models                    # Model listing
```

Response includes `prompt_token_ids` (top-level) and `token_ids` (per-choice) when `return_token_ids=True`.

## Implementation Order

1. `compat.py` — v5 compatibility (test immediately)
2. `convert_checkpoint.py` — export training checkpoint
3. `masks.py` — decode + compaction masks
4. `sampling.py` — sampling utilities
5. `config.py` — InferenceConfig
6. `engine.py` — CompactionInferenceEngine
7. Logit equivalence test vs `blockwise_forward_eval()`
8. `server.py` — FastAPI server
9. API compatibility test
10. GSM8K integration test via verifiers

## Key Training Code Reused

| Module | Functions |
|--------|-----------|
| `attention.py` | `build_4d_attention_mask()`, `forward_layers()` |
| `model.py` | `embed_tokens()`, `get_rotary_embeddings()`, `get_lm_head()`, `get_inner_model()` |
| `kv_manager.py` | `extract_compact_kv()`, `create_cache_with_compact_kv()` |

## Environment Setup

```bash
UV_CACHE_DIR=/pscratch/sd/s/siddart2/uv-cache uv venv .venv --python 3.12
source .venv/bin/activate
UV_CACHE_DIR=/pscratch/sd/s/siddart2/uv-cache uv pip install \
    "transformers>=5.4.0" "peft>=0.18.0" "torch>=2.9.0" \
    accelerate "fastapi[standard]" uvicorn openai math-verify datasets
```

## GSM8K Integration Test

```bash
# Convert checkpoint
python -m src.inference.convert_checkpoint --ckpt <path> --base-model Qwen/Qwen3-0.6B-Base --output <dir>

# Start server
python -m src.inference.server --model Qwen/Qwen3-0.6B-Base --adapter <adapter_dir> \
    --compaction-params <params.pt> --W 512 --P 64 --port 8000

# Evaluate
vf-eval math-env -a '{"dataset_name":"openai/gsm8k","dataset_subset":"main"}' \
    -m compaction-model -b http://localhost:8000/v1 -n 200 -t 2048
```

## Adversarial Findings (Resolved)

| Finding | Resolution |
|---------|------------|
| Partial block compaction | Process WITHOUT compaction, grow until W |
| v5 compatibility | Compat layer: handle tuple returns, verify DynamicCache API |
| Logprob off-by-one | Explicitly track previous-step logits |
| Non-standard API fields | Include `prompt_token_ids` + `token_ids` |
| Weight update schema | `weight_dir` not `weight_path` |
| GPU concurrency | asyncio.Lock serialization |
| Missing /tokenize | Added to endpoint list |
