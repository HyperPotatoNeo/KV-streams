#!/bin/bash
# Condition A: 2-node (8 GPU) DDP, full context
# Total batch = 8 GPU * 2 batch * 32 accum = 512
# Master node: $1, worker node: $2
set -e
source /pscratch/sd/s/siddart2/prime-rl/.venv/bin/activate
cd /pscratch/sd/s/siddart2/kv-self-compaction-phase2
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_DATASETS_CACHE=/pscratch/sd/s/siddart2/hf_cache
export HF_HOME=/pscratch/sd/s/siddart2/hf_cache
export WANDB_API_KEY=595199cad0de28f309ce22cb212dcbeeb21b06d8

MASTER=$1
NODE_RANK=$2

echo "Starting A 2-node DDP: master=$MASTER node_rank=$NODE_RANK at $(date) on $(hostname)"
torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER \
  --master_port=29500 \
  -m src.train \
  --condition A \
  --max_steps 600 \
  --eval_every 100 \
  --save_every 50 \
  --batch_size 2 \
  --grad_accum 32 \
  --lr 1e-4 \
  --max_examples_per_dataset 50000 \
  --output_dir outputs/ddp_scaleup_W512 2>&1
echo "Finished A at $(date)"
