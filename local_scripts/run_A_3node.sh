#!/bin/bash
# Condition A: 3-node (12 GPU) DDP, full context
# Total batch = 12 GPU * 1 batch * 43 accum ≈ 516 (close to 512)
set -e
source /pscratch/sd/s/siddart2/prime-rl/.venv/bin/activate
cd /pscratch/sd/s/siddart2/kv-self-compaction-phase2
export PYTHONPATH=$PWD:$PYTHONPATH
export HF_DATASETS_CACHE=/pscratch/sd/s/siddart2/hf_cache
export HF_HOME=/pscratch/sd/s/siddart2/hf_cache
export WANDB_API_KEY=595199cad0de28f309ce22cb212dcbeeb21b06d8

MASTER=$1
NODE_RANK=$2

# NCCL and error handling
export NCCL_TIMEOUT=7200000  # 2 hours
export TORCH_NCCL_BLOCKING_WAIT=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reduce fragmentation OOM
export TORCHELASTIC_ERROR_FILE=/pscratch/sd/s/siddart2/kv-self-compaction-phase2/logs/A_error_node${NODE_RANK}.json

echo "Starting A 3-node DDP: master=$MASTER node_rank=$NODE_RANK at $(date) on $(hostname)"
torchrun \
  --nnodes=3 \
  --nproc_per_node=4 \
  --node_rank=$NODE_RANK \
  --master_addr=$MASTER \
  --master_port=29504 \
  --redirects 3 \
  --log_dir /pscratch/sd/s/siddart2/kv-self-compaction-phase2/logs/torchrun_A \
  -m src.train \
  --condition A \
  --W 512 \
  --P 64 \
  --K 8 \
  --max_steps 600 \
  --eval_every 100 \
  --save_every 50 \
  --batch_size 1 \
  --grad_accum 43 \
  --lr 1e-4 \
  --max_examples_per_dataset 50000 \
  --output_dir outputs/ddp_scaleup_W512 2>&1
echo "Finished A node_rank=$NODE_RANK at $(date)"
