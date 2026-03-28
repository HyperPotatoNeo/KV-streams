#!/bin/bash
set -e
source /pscratch/sd/s/siddart2/prime-rl/.venv/bin/activate
UV_CACHE_DIR=/pscratch/sd/s/siddart2/uv-cache uv pip install peft>=0.14.0
python -c "import peft; print(f'peft={peft.__version__} installed OK')"
