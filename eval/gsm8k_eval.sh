#!/bin/bash
# GSM8K evaluation with KV Self-Compaction inference
# Run on a GPU node via srun --overlap
set -e
cd /pscratch/sd/s/siddart2/kv-self-compaction-phase2
export PYTHONPATH=.
export HF_HOME=/pscratch/sd/s/siddart2/hf_cache
source .venv/bin/activate

MODEL=${1:-"Qwen/Qwen3-0.6B-Base"}
ADAPTER=${2:-"outputs/ddp_scaleup_W512/condition_B/inference/adapter"}
COMPACTION_PARAMS=${3:-"outputs/ddp_scaleup_W512/condition_B/inference/compaction_params.pt"}
W=${4:-512}
P=${5:-64}
N_EXAMPLES=${6:-50}
PORT=${7:-8123}

echo "=== GSM8K Evaluation ==="
echo "Model: $MODEL"
echo "Adapter: $ADAPTER"
echo "W=$W P=$P"
echo "Examples: $N_EXAMPLES"
echo "Port: $PORT"
echo ""

# Start server in background
echo "Starting inference server..."
python -m src.inference.server \
    --model "$MODEL" \
    --adapter "$ADAPTER" \
    --compaction-params "$COMPACTION_PARAMS" \
    --W "$W" --P "$P" \
    --host 0.0.0.0 --port "$PORT" &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server startup..."
for i in $(seq 1 60); do
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo "Server ready after ${i}s"
        break
    fi
    sleep 1
done

if ! curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
    echo "ERROR: Server did not start within 60s"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

# Run GSM8K evaluation
echo ""
echo "=== Running GSM8K eval ($N_EXAMPLES examples) ==="
python -c "
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url='http://localhost:$PORT/v1', api_key='none')

# Load GSM8K dataset
from datasets import load_dataset
ds = load_dataset('openai/gsm8k', 'main', split='test')
examples = list(ds.select(range($N_EXAMPLES)))

import re

def extract_boxed(text):
    \"\"\"Extract answer from \\boxed{...}\"\"\"
    matches = re.findall(r'\\\\boxed\{([^}]*)\}', text)
    if matches:
        return matches[-1].strip()
    return None

def extract_gsm8k_answer(answer_str):
    \"\"\"Extract numeric answer from GSM8K answer string (after ####)\"\"\"
    parts = answer_str.split('####')
    if len(parts) > 1:
        return parts[-1].strip().replace(',', '')
    return answer_str.strip()

async def evaluate_one(example, idx):
    question = example['question']
    gold = extract_gsm8k_answer(example['answer'])

    try:
        response = await client.chat.completions.create(
            model='compaction-model',
            messages=[{
                'role': 'user',
                'content': f'Solve the following math problem. Put the final answer in \\\\boxed{{}}.\\n\\n{question}'
            }],
            max_tokens=1024,
            temperature=0.0,
        )
        text = response.choices[0].message.content
        pred = extract_boxed(text)

        # Normalize for comparison
        if pred is not None:
            pred_clean = pred.replace(',', '').strip()
            correct = pred_clean == gold
        else:
            correct = False

        if idx < 5:  # Show first 5
            print(f'  [{idx}] Gold={gold}, Pred={pred}, Correct={correct}')
            if not correct:
                print(f'       Text: {text[:150]}...')

        return correct
    except Exception as e:
        print(f'  [{idx}] ERROR: {e}')
        return False

async def main():
    tasks = [evaluate_one(ex, i) for i, ex in enumerate(examples)]
    results = await asyncio.gather(*tasks)

    correct = sum(results)
    total = len(results)
    accuracy = correct / total * 100

    print(f'\\n=== Results ===')
    print(f'Correct: {correct}/{total} = {accuracy:.1f}%')
    return accuracy

accuracy = asyncio.run(main())
"

echo ""
echo "Evaluation complete."

# Cleanup
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null
echo "Server stopped."
