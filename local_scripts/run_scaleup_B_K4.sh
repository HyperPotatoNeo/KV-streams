#!/bin/bash
set -e
source /pscratch/sd/s/siddart2/prime-rl/.venv/bin/activate
cd /pscratch/sd/s/siddart2/kv-self-compaction-phase2
export PYTHONPATH=/pscratch/sd/s/siddart2/kv-self-compaction-phase2:$PYTHONPATH
export HF_DATASETS_CACHE=/pscratch/sd/s/siddart2/hf_cache
export HF_HOME=/pscratch/sd/s/siddart2/hf_cache

echo "Starting B with K=4 (deeper BPTT) at $(date)"
CUDA_VISIBLE_DEVICES=1 python -m src.train \
  --condition B \
  --K 4 \
  --max_steps 660 \
  --eval_every 100 \
  --save_every 50 \
  --batch_size 4 \
  --grad_accum 4 \
  --max_examples_per_dataset 50000 \
  --output_dir outputs/scaleup_K4 2>&1
echo "Finished at $(date)"
