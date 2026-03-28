#!/bin/bash
# Condition B: 3-node (12 GPU) DDP, W=512 P=64 K=8
# Total batch = 12 GPU * 2 batch * 22 accum ≈ 528 (close to 512)
# ~3x faster than single node
set -e
source /pscratch/sd/s/siddart2/prime-rl/.venv/bin/activate
cd /pscratch/sd/s/siddart2/kv-self-compaction-phase2
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_DATASETS_CACHE=/pscratch/sd/s/siddart2/hf_cache
export HF_HOME=/pscratch/sd/s/siddart2/hf_cache
export WANDB_API_KEY=595199cad0de28f309ce22cb212dcbeeb21b06d8

MASTER=$1
NODE_RANK=$2

export NCCL_TIMEOUT=3600000  # 1 hour timeout (eval on rank 0 takes long, others wait)
export TORCH_NCCL_BLOCKING_WAIT=0

echo "Starting B 3-node DDP: master=$MASTER node_rank=$NODE_RANK at $(date) on $(hostname)"
torchrun \
  --nnodes=3 \
  --nproc_per_node=4 \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER \
  --master_port=29503 \
  -m src.train \
  --condition B \
  --W 512 \
  --P 64 \
  --K 8 \
  --max_steps 600 \
  --eval_every 100 \
  --save_every 50 \
  --batch_size 2 \
  --grad_accum 22 \
  --lr 1e-4 \
  --max_examples_per_dataset 50000 \
  --output_dir outputs/ddp_scaleup_W512 2>&1
echo "Finished B node_rank=$NODE_RANK at $(date)"
