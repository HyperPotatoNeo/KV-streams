#!/bin/bash
set -e
source /pscratch/sd/s/siddart2/prime-rl/.venv/bin/activate
cd /pscratch/sd/s/siddart2/kv-self-compaction-phase2
export PYTHONPATH=/pscratch/sd/s/siddart2/kv-self-compaction-phase2:$PYTHONPATH
export HF_DATASETS_CACHE=/pscratch/sd/s/siddart2/hf_cache
export HF_HOME=/pscratch/sd/s/siddart2/hf_cache

CUDA_VISIBLE_DEVICES=0 python -m src.train \
  --condition B \
  --max_steps 10 \
  --eval_every 5 \
  --batch_size 2 \
  --max_examples_per_dataset 100 \
  --output_dir outputs/smoke_test 2>&1
