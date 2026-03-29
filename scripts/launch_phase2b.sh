#!/bin/bash
# Phase 2b: 4 GPUs, all conditions
# Usage:
#   salloc --reservation=_CAP_tinker -A m5017 -C "gpu&hbm80g" --gpus-per-node=4 --nodes=1 --time=8:00:00
#   bash scripts/launch_phase2b.sh
set -e

export HOME=$SCRATCH
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
cd $SCRATCH

# Run each condition on a separate GPU
for CONDITION in A B C D; do
  GPU_ID=$(($(echo $CONDITION | tr 'ABCD' '0123')))
  echo "Launching condition $CONDITION on GPU $GPU_ID"
  podman-hpc run --rm -d \
    --user "$(id -u):$(id -g)" --name kv-phase2-$CONDITION \
    --group-add keep-groups --userns keep-id --gpu --nccl --shm-size=8g \
    -e SCRATCH -e HOME -e CUDA_VISIBLE_DEVICES=$GPU_ID \
    -e UV_CACHE_DIR=$SCRATCH/uv-cache \
    -v "$SCRATCH":"$SCRATCH" -v /global/homes/s/siddart2:/global/homes/s/siddart2 \
    -w "$SCRATCH/kv-self-compaction-phase2" \
    docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 \
    bash -c "unset NCCL_SOCKET_IFNAME && uv pip install peft datasets && python -m src.train --condition $CONDITION --max_examples_per_dataset 50000"
done

echo "All 4 conditions launched. Monitor with: podman-hpc logs -f kv-phase2-B"
