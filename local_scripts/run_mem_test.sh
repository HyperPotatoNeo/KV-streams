#!/bin/bash
set -e
source /pscratch/sd/s/siddart2/prime-rl/.venv/bin/activate
cd /pscratch/sd/s/siddart2/kv-self-compaction-phase2
export PYTHONPATH=/pscratch/sd/s/siddart2/kv-self-compaction-phase2:$PYTHONPATH
export HF_DATASETS_CACHE=/pscratch/sd/s/siddart2/hf_cache
export HF_HOME=/pscratch/sd/s/siddart2/hf_cache

echo "Testing batch_size=8..."
CUDA_VISIBLE_DEVICES=0 python -m src.train \
  --condition B \
  --max_steps 2 \
  --eval_every 999 \
  --save_every 999 \
  --batch_size 8 \
  --grad_accum 1 \
  --lr 1e-4 \
  --max_examples_per_dataset 100 \
  --output_dir outputs/mem_test 2>&1 | tail -5

echo "---"
echo "Testing batch_size=16..."
CUDA_VISIBLE_DEVICES=1 python -m src.train \
  --condition B \
  --max_steps 2 \
  --eval_every 999 \
  --save_every 999 \
  --batch_size 16 \
  --grad_accum 1 \
  --lr 1e-4 \
  --max_examples_per_dataset 100 \
  --output_dir outputs/mem_test2 2>&1 | tail -5
