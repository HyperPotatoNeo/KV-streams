#!/bin/bash
# Condition B with K=1: compact_kv flows but no gradient through time
# 3-node (12 GPU) DDP, W=512 P=64 K=1
# Total batch = 12 GPU * 2 batch * 22 accum ≈ 528
set -e
source /pscratch/sd/s/siddart2/prime-rl/.venv/bin/activate
cd /pscratch/sd/s/siddart2/kv-self-compaction-phase2
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_DATASETS_CACHE=/pscratch/sd/s/siddart2/hf_cache
export HF_HOME=/pscratch/sd/s/siddart2/hf_cache
export WANDB_API_KEY=595199cad0de28f309ce22cb212dcbeeb21b06d8
export NCCL_TIMEOUT=7200000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MASTER=$1
NODE_RANK=$2

echo "Starting B K=1 3-node DDP: master=$MASTER node_rank=$NODE_RANK at $(date) on $(hostname)"
torchrun \
  --nnodes=3 \
  --nproc_per_node=4 \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER \
  --master_port=29505 \
  -m src.train \
  --condition B \
  --W 512 \
  --P 64 \
  --K 1 \
  --max_steps 600 \
  --eval_every 100 \
  --save_every 50 \
  --batch_size 2 \
  --grad_accum 22 \
  --lr 1e-4 \
  --max_examples_per_dataset 50000 \
  --output_dir outputs/ddp_scaleup_W512_K1 2>&1
echo "Finished B K=1 node_rank=$NODE_RANK at $(date)"
