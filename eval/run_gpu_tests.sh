#!/bin/bash
# Run GPU tests for KV Self-Compaction inference engine
# Must be run on a compute node with GPU access
#
# Usage from login node:
#   srun --jobid=JOBID --overlap bash eval/run_gpu_tests.sh > logs/gpu_tests.log 2>&1 &

set -e
cd /pscratch/sd/s/siddart2/kv-self-compaction-phase2

export PYTHONPATH=.
export HF_HOME=/pscratch/sd/s/siddart2/hf_cache

echo "=== GPU Test Suite for KV Self-Compaction Inference ==="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# Use the project venv
source .venv/bin/activate
echo "Python: $(python --version)"
echo "Transformers: $(python -c 'import transformers; print(transformers.__version__)')"
echo ""

# 1. Run CPU tests first (sanity check)
echo "=== Step 1: CPU Tests ==="
python -m pytest tests/test_inference/test_masks.py tests/test_inference/test_server.py -v
echo ""

# 2. Run GPU engine tests
echo "=== Step 2: GPU Engine Tests ==="
python -m pytest tests/test_inference/test_engine.py -v -x
echo ""

# 3. Quick generation test
echo "=== Step 3: Quick Generation Test ==="
python -c "
import torch
from src.inference.engine import CompactionInferenceEngine
import torch.nn as nn

print('Loading engine on GPU...')
engine = CompactionInferenceEngine(
    base_model_name='Qwen/Qwen3-0.6B-Base',
    W=128, P=16,
    device='cuda', dtype_str='bfloat16',
)

engine.model.compaction_embeddings = nn.Parameter(
    torch.randn(16, 1024, device='cuda', dtype=torch.bfloat16) * 0.02,
    requires_grad=False,
)
engine.model.compact_attn_bias = nn.Parameter(
    torch.full((16,), -2.0, device='cuda', dtype=torch.bfloat16),
    requires_grad=False,
)

# Generate with compaction trigger
prompt = engine.tokenizer.encode('Solve: What is 15 * 7? Think step by step.', add_special_tokens=True)
print(f'Prompt: {len(prompt)} tokens')

result = engine.generate(
    prompt, max_new_tokens=200, temperature=0.0, return_logprobs=True
)
print(f'Generated: {len(result.token_ids)} tokens')
print(f'Finish: {result.finish_reason}')
print(f'Text: {result.text[:200]}')
print(f'Logprobs range: [{min(result.logprobs):.3f}, {max(result.logprobs):.3f}]')
print('Generation test passed!')
"

echo ""
echo "=== All GPU Tests Complete ==="
echo "Date: $(date)"
