#!/bin/bash
# Condition C: truncation to last W=512 tokens
# Total batch = 4 GPU * 4 batch * 32 accum = 512
# (C only processes 512 tokens, standard forward — less memory than blockwise)
set -e
source /pscratch/sd/s/siddart2/prime-rl/.venv/bin/activate
cd /pscratch/sd/s/siddart2/kv-self-compaction-phase2
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_DATASETS_CACHE=/pscratch/sd/s/siddart2/hf_cache
export HF_HOME=/pscratch/sd/s/siddart2/hf_cache
export WANDB_API_KEY=595199cad0de28f309ce22cb212dcbeeb21b06d8

echo "Starting condition C W=512 (truncation) at $(date) on $(hostname)"
torchrun --standalone --nproc_per_node=4 -m src.train \
  --condition C \
  --W 512 \
  --P 64 \
  --K 8 \
  --max_steps 600 \
  --eval_every 100 \
  --save_every 50 \
  --batch_size 4 \
  --grad_accum 32 \
  --lr 1e-4 \
  --max_examples_per_dataset 50000 \
  --output_dir outputs/ddp_scaleup_W512 2>&1
echo "Finished condition C at $(date)"
