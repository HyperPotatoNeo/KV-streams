#!/bin/bash
# Phase 2a: Single GPU interactive session for KV Self-Compaction
# Usage:
#   1. Allocate a GPU node first:
#      salloc --reservation=_CAP_tinker -A m5017 -C "gpu&hbm80g" --gpus-per-node=4 --nodes=1 --time=4:00:00
#   2. Then run this script on the allocated node:
#      bash scripts/launch_phase2a.sh
set -e

export HOME=$SCRATCH
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
cd $SCRATCH

podman-hpc run --rm -it \
  --user "$(id -u):$(id -g)" --replace --name kv-phase2 \
  --group-add keep-groups --userns keep-id --gpu --nccl --shm-size=8g \
  -e SCRATCH -e HOME -e CUDA_VISIBLE_DEVICES=0 \
  -e UV_CACHE_DIR=$SCRATCH/uv-cache \
  -v "$SCRATCH":"$SCRATCH" -v /global/homes/s/siddart2:/global/homes/s/siddart2 \
  -w "$SCRATCH/kv-self-compaction-phase2" \
  docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 \
  bash -c 'unset NCCL_SOCKET_IFNAME && uv pip install peft datasets && python -m src.train --condition B'
