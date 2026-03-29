#!/bin/bash
set -e
cd /pscratch/sd/s/siddart2/kv-self-compaction-phase2
export PYTHONPATH=.
export HF_HOME=/pscratch/sd/s/siddart2/hf_cache

echo "=== GPU Tests ==="
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null)"

source .venv/bin/activate

# 1. Quick sanity: mask tests
echo ""
echo "=== Mask Tests (CPU) ==="
python -m pytest tests/test_inference/test_masks.py -q

# 2. Engine GPU tests
echo ""
echo "=== Engine GPU Tests ==="
python -m pytest tests/test_inference/test_engine.py -v -x 2>&1

# 3. Quick generation test
echo ""
echo "=== Generation Test ==="
python -c "
import torch, torch.nn as nn
from src.inference.engine import CompactionInferenceEngine

engine = CompactionInferenceEngine(
    base_model_name='Qwen/Qwen3-0.6B-Base',
    W=128, P=16, device='cuda', dtype_str='bfloat16',
)
engine.model.compaction_embeddings = nn.Parameter(
    torch.randn(16, 1024, device='cuda', dtype=torch.bfloat16) * 0.02, requires_grad=False)
engine.model.compact_attn_bias = nn.Parameter(
    torch.full((16,), -2.0, device='cuda', dtype=torch.bfloat16), requires_grad=False)

prompt = engine.tokenizer.encode('What is 15 * 7? Think step by step.', add_special_tokens=True)
print(f'Prompt: {len(prompt)} tokens')
result = engine.generate(prompt, max_new_tokens=200, temperature=0.0, return_logprobs=True)
print(f'Generated: {len(result.token_ids)} tokens, finish={result.finish_reason}')
print(f'Text: {result.text[:300]}')
print(f'Logprobs: min={min(result.logprobs):.3f}, max={max(result.logprobs):.3f}')
print('PASS')
"

echo ""
echo "=== All GPU Tests Complete ==="
