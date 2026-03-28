#!/bin/bash
# Quick DDP smoke test: 5 steps, 4 GPUs
set -e
source /pscratch/sd/s/siddart2/prime-rl/.venv/bin/activate
cd /pscratch/sd/s/siddart2/kv-self-compaction-phase2
export PYTHONPATH=/pscratch/sd/s/siddart2/kv-self-compaction-phase2:$PYTHONPATH
export HF_DATASETS_CACHE=/pscratch/sd/s/siddart2/hf_cache
export HF_HOME=/pscratch/sd/s/siddart2/hf_cache

echo "DDP smoke test: 4 GPUs, 5 steps at $(date)"
torchrun --standalone --nproc_per_node=4 -m src.train \
  --condition B \
  --max_steps 5 \
  --eval_every 5 \
  --save_every 999 \
  --batch_size 8 \
  --grad_accum 2 \
  --lr 1e-4 \
  --max_examples_per_dataset 200 \
  --output_dir outputs/ddp_smoke 2>&1
echo "Finished at $(date)"
