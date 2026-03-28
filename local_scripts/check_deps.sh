#!/bin/bash
set -e
source /pscratch/sd/s/siddart2/prime-rl/.venv/bin/activate
cd /pscratch/sd/s/siddart2/kv-self-compaction-phase2

python -c "
import torch
print(f'torch={torch.__version__}, GPUs={torch.cuda.device_count()}')
import transformers
print(f'transformers={transformers.__version__}')
try:
    import peft
    print(f'peft={peft.__version__}')
except ImportError:
    print('peft NOT installed')
try:
    import datasets
    print(f'datasets={datasets.__version__}')
except ImportError:
    print('datasets NOT installed')
try:
    import pytest
    print(f'pytest={pytest.__version__}')
except ImportError:
    print('pytest NOT installed')
try:
    import wandb
    print(f'wandb={wandb.__version__}')
except ImportError:
    print('wandb NOT installed')
print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')
"
