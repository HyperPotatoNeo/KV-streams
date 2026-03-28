#!/bin/bash
set -e
echo "Hostname: $(hostname)"
echo "Checking GPU access..."
source /pscratch/sd/s/siddart2/prime-rl/.venv/bin/activate
python -c "import torch; print(f'GPUs: {torch.cuda.device_count()}, Name: {torch.cuda.get_device_name(0)}')"
echo "Node2 working"
