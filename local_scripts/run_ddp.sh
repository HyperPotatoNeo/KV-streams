#!/bin/bash
# Usage: bash run_ddp.sh <CONDITION> [extra args]
# Launches torchrun with 4 GPUs, total batch size 512 (4 GPU * 8 batch * 16 accum)
set -e
source /pscratch/sd/s/siddart2/prime-rl/.venv/bin/activate
cd /pscratch/sd/s/siddart2/kv-self-compaction-phase2
export PYTHONPATH=/pscratch/sd/s/siddart2/kv-self-compaction-phase2:$PYTHONPATH
export HF_DATASETS_CACHE=/pscratch/sd/s/siddart2/hf_cache
export HF_HOME=/pscratch/sd/s/siddart2/hf_cache
export WANDB_API_KEY=595199cad0de28f309ce22cb212dcbeeb21b06d8

CONDITION=${1:-B}
shift || true

# Find latest checkpoint for resume
OUTPUT_DIR="${OUTPUT_DIR:-outputs/ddp_scaleup}"
CKPT_DIR="$OUTPUT_DIR/condition_${CONDITION}"
RESUME_ARG=""
if [ -d "$CKPT_DIR" ]; then
  LATEST=$(ls -d ${CKPT_DIR}/checkpoint-*/checkpoint.pt 2>/dev/null | sort -t- -k2 -n | tail -1)
  if [ -n "$LATEST" ]; then
    echo "Resuming from: $LATEST"
    RESUME_ARG="--resume_from $LATEST"
  fi
fi

echo "Starting condition $CONDITION (4-GPU DDP, bs_total=512) at $(date) on $(hostname)"
torchrun --standalone --nproc_per_node=4 -m src.train \
  --condition $CONDITION \
  --max_steps 600 \
  --eval_every 100 \
  --save_every 50 \
  --batch_size 8 \
  --grad_accum 16 \
  --lr 1e-4 \
  --max_examples_per_dataset 50000 \
  --output_dir "$OUTPUT_DIR" \
  $RESUME_ARG \
  "$@" 2>&1
echo "Finished condition $CONDITION at $(date)"
