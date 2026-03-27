#!/bin/bash
set -e
export HOME=/pscratch/sd/s/siddart2
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
cd $HOME

podman-hpc run --rm -d \
  --user "$(id -u):$(id -g)" --replace --name kv-phase2 \
  --group-add keep-groups --userns keep-id --gpu --nccl --shm-size=8g \
  -e SCRATCH=/pscratch/sd/s/siddart2 -e HOME=/pscratch/sd/s/siddart2 \
  -e UV_CACHE_DIR=/pscratch/sd/s/siddart2/uv-cache \
  -e HF_DATASETS_CACHE=/pscratch/sd/s/siddart2/hf_cache \
  -e HF_HOME=/pscratch/sd/s/siddart2/hf_cache \
  -v "/pscratch/sd/s/siddart2":"/pscratch/sd/s/siddart2" \
  -v "/global/homes/s/siddart2":"/global/homes/s/siddart2" \
  -w "/pscratch/sd/s/siddart2/kv-self-compaction-phase2" \
  docker.io/novaskyai/skyrl-train-ray-2.51.1-py3.12-cu12.8 \
  bash -c 'unset NCCL_SOCKET_IFNAME && sleep infinity'

echo "Container kv-phase2 started on $(hostname)"
