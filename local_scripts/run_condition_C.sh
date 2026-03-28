#!/bin/bash
set -e
source /pscratch/sd/s/siddart2/prime-rl/.venv/bin/activate
cd /pscratch/sd/s/siddart2/kv-self-compaction-phase2
export PYTHONPATH=/pscratch/sd/s/siddart2/kv-self-compaction-phase2:$PYTHONPATH
export HF_DATASETS_CACHE=/pscratch/sd/s/siddart2/hf_cache
export HF_HOME=/pscratch/sd/s/siddart2/hf_cache

CUDA_VISIBLE_DEVICES=1 python -m src.train \
  --condition C \
  --max_steps 200 \
  --eval_every 50 \
  --batch_size 4 \
  --max_examples_per_dataset 5000 \
  --output_dir outputs/phase2a_v2 2>&1
