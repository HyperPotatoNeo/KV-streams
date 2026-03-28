#!/bin/bash
# W=512, P=64, K=8 for blockwise conditions (B/D/E)
# Total batch = 4 GPU * 2 batch * 64 accum = 512
# (batch_size=4 OOMs with W=512 K=8 eager attention)
set -e
source /pscratch/sd/s/siddart2/prime-rl/.venv/bin/activate
cd /pscratch/sd/s/siddart2/kv-self-compaction-phase2
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_DATASETS_CACHE=/pscratch/sd/s/siddart2/hf_cache
export HF_HOME=/pscratch/sd/s/siddart2/hf_cache
export WANDB_API_KEY=595199cad0de28f309ce22cb212dcbeeb21b06d8

CONDITION=${1:-B}
shift || true

# Find latest checkpoint for resume
CKPT_DIR="outputs/ddp_scaleup_W512/condition_${CONDITION}"
RESUME_ARG=""
if [ -d "$CKPT_DIR" ]; then
  LATEST=$(ls -d ${CKPT_DIR}/checkpoint-*/checkpoint.pt 2>/dev/null | sort -t- -k2 -n | tail -1)
  if [ -n "$LATEST" ]; then
    echo "Resuming from: $LATEST"
    RESUME_ARG="--resume_from $LATEST"
  fi
fi

echo "Starting condition $CONDITION W=512 P=64 K=8 (4-GPU DDP, bs=2) at $(date) on $(hostname)"
torchrun --standalone --nproc_per_node=4 -m src.train \
  --condition $CONDITION \
  --W 512 \
  --P 64 \
  --K 8 \
  --max_steps 600 \
  --eval_every 100 \
  --save_every 50 \
  --batch_size 2 \
  --grad_accum 64 \
  --lr 1e-4 \
  --max_examples_per_dataset 50000 \
  --output_dir outputs/ddp_scaleup_W512 \
  $RESUME_ARG \
  "$@" 2>&1
echo "Finished condition $CONDITION at $(date)"
