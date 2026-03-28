#!/bin/bash
# Condition A: full context needs smaller per-GPU batch (OOM with batch_size=8)
# Total batch = 4 GPU * 2 batch * 64 accum = 512
set -e
source /pscratch/sd/s/siddart2/prime-rl/.venv/bin/activate
cd /pscratch/sd/s/siddart2/kv-self-compaction-phase2
export PYTHONPATH=/pscratch/sd/s/siddart2/kv-self-compaction-phase2:$PYTHONPATH
export HF_DATASETS_CACHE=/pscratch/sd/s/siddart2/hf_cache
export HF_HOME=/pscratch/sd/s/siddart2/hf_cache

echo "Starting condition A (4-GPU DDP, bs_total=512) at $(date) on $(hostname)"
torchrun --standalone --nproc_per_node=4 -m src.train \
  --condition A \
  --max_steps 600 \
  --eval_every 100 \
  --save_every 50 \
  --batch_size 2 \
  --grad_accum 64 \
  --lr 1e-4 \
  --max_examples_per_dataset 50000 \
  --output_dir outputs/ddp_scaleup 2>&1
echo "Finished condition A at $(date)"
