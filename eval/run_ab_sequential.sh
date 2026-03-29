#!/bin/bash
# Run Condition A then B sequentially on same node
set -e
cd /pscratch/sd/s/siddart2/kv-self-compaction-phase2
export PYTHONPATH=.
export HF_HOME=/pscratch/sd/s/siddart2/hf_cache
export NCCL_DEBUG=WARN
source .venv/bin/activate

N=${1:-50}
MAX_TOK=${2:-4096}

echo "=== Node: $(hostname), GPUs: $(nvidia-smi -L | wc -l) ==="
echo "Examples: $N, Max tokens: $MAX_TOK"
echo ""

echo "========== CONDITION A (full context) =========="
torchrun --nproc_per_node=4 --master_port=29991 eval/gsm8k_condition_a.py $N $MAX_TOK 2>&1
echo ""

echo "========== CONDITION B (compaction) =========="
torchrun --nproc_per_node=4 --master_port=29992 eval/quick_gsm8k_parallel.py $N $MAX_TOK 2>&1
echo ""

echo "========== BOTH CONDITIONS COMPLETE =========="
