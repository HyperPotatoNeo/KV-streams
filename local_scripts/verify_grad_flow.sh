#!/bin/bash
set -e
source /pscratch/sd/s/siddart2/prime-rl/.venv/bin/activate
cd /pscratch/sd/s/siddart2/kv-self-compaction-phase2
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_DATASETS_CACHE=/pscratch/sd/s/siddart2/hf_cache
export HF_HOME=/pscratch/sd/s/siddart2/hf_cache

CUDA_VISIBLE_DEVICES=0 python -c "
import torch
from src.config import CompactionConfig
from src.model import setup_model
from src.blockwise import blockwise_train_step

config = CompactionConfig(W=32, P=4, K=2, max_seq_len=128, batch_size=1)
model = setup_model(config).cuda()

# Synthetic data
B, S = 1, 128
input_ids = torch.randint(0, 1000, (B, S)).cuda()
labels = torch.randint(0, 1000, (B, S)).cuda()
attention_mask = torch.ones(B, S, dtype=torch.long).cuda()

# Zero grads
for p in model.parameters():
    if p.grad is not None:
        p.grad.zero_()

# Forward + backward
loss = blockwise_train_step(model, input_ids, labels, attention_mask, config)

# Check gradients
embed_grad = model.compaction_embeddings.grad
bias_grad = model.compact_attn_bias.grad

print(f'Loss: {loss:.4f}')
print(f'compaction_embeddings.grad: {\"NONE\" if embed_grad is None else f\"max={embed_grad.abs().max():.6f} mean={embed_grad.abs().mean():.6f}\"}')
print(f'compact_attn_bias.grad: {\"NONE\" if bias_grad is None else f\"values={bias_grad.tolist()}\"}')
print(f'compact_attn_bias.data: {model.compact_attn_bias.data.tolist()}')

# Check a few LoRA params
lora_grads = 0
lora_total = 0
for name, p in model.named_parameters():
    if 'lora' in name.lower() and p.grad is not None:
        lora_grads += 1
        lora_total += 1
    elif 'lora' in name.lower():
        lora_total += 1
print(f'LoRA params with grad: {lora_grads}/{lora_total}')
" 2>&1
