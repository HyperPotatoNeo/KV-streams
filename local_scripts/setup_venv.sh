#!/bin/bash
set -e
cd /pscratch/sd/s/siddart2/kv-self-compaction-phase2

PYTHON=/pscratch/sd/s/siddart2/prime-rl/.venv/bin/python3.12

# Create venv using the existing Python 3.12
$PYTHON -m venv .venv --system-site-packages

# Activate
source .venv/bin/activate

# Check python version
python --version
python -c "import torch; print(f'torch={torch.__version__}, GPUs={torch.cuda.device_count()}')"

# Install project-specific deps (peft, datasets should already be available from system)
UV_CACHE_DIR=/pscratch/sd/s/siddart2/uv-cache uv pip install peft datasets wandb 2>&1 | tail -5

# Verify imports
python -c "
import torch, transformers, peft, datasets
print(f'transformers={transformers.__version__}')
print(f'peft={peft.__version__}')
print(f'datasets={datasets.__version__}')
print('All imports OK')
"

echo "Setup complete!"
