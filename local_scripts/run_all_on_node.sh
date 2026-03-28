#!/bin/bash
# Run conditions B+D on 2 GPUs of this node
set -e
source /pscratch/sd/s/siddart2/prime-rl/.venv/bin/activate
cd /pscratch/sd/s/siddart2/kv-self-compaction-phase2
export PYTHONPATH=/pscratch/sd/s/siddart2/kv-self-compaction-phase2:$PYTHONPATH
export HF_DATASETS_CACHE=/pscratch/sd/s/siddart2/hf_cache
export HF_HOME=/pscratch/sd/s/siddart2/hf_cache

CONDITION=$1
GPU=$2

echo "Starting condition $CONDITION on GPU $GPU at $(date)"
CUDA_VISIBLE_DEVICES=$GPU python -m src.train \
  --condition $CONDITION \
  --max_steps 200 \
  --eval_every 50 \
  --batch_size 4 \
  --max_examples_per_dataset 5000 \
  --output_dir outputs/phase2a_v2 2>&1
echo "Finished condition $CONDITION at $(date)"
