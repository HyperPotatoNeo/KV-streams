#!/bin/bash
export HOME=/pscratch/sd/s/siddart2
export PODMANHPC_PODMAN_BIN=/global/common/shared/das/podman-4.7.0/bin/podman
podman-hpc exec kv-phase2 bash -c 'source /pscratch/sd/s/siddart2/prime-rl/.venv/bin/activate && python --version && python -c "import torch; print(f\"GPUs: {torch.cuda.device_count()}\")"'
