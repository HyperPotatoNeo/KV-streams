"""Quick GSM8K eval — runs directly without server overhead."""

import re
import sys
import time
import torch
from src.inference.engine import CompactionInferenceEngine


def extract_boxed(text):
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    if matches:
        return matches[-1].strip()
    return None


def extract_gsm8k_answer(answer_str):
    parts = answer_str.split('####')
    if len(parts) > 1:
        return parts[-1].strip().replace(',', '')
    return answer_str.strip()


def main():
    n_examples = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 512

    print(f"Loading compaction model (W=512, P=64)...")
    engine = CompactionInferenceEngine(
        base_model_name="Qwen/Qwen3-0.6B-Base",
        adapter_path="outputs/ddp_scaleup_W512/condition_B/inference/adapter",
        compaction_params_path="outputs/ddp_scaleup_W512/condition_B/inference/compaction_params.pt",
        W=512, P=64, device="cuda", dtype_str="bfloat16",
    )
    print(f"Bias: {engine.model.compact_attn_bias.data.mean():.3f}")

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    examples = list(ds.select(range(n_examples)))

    correct = 0
    total = 0
    t_start = time.time()

    for i, ex in enumerate(examples):
        question = ex["question"]
        gold = extract_gsm8k_answer(ex["answer"])

        prompt = engine.tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Solve the following math problem. Put the final answer in \\boxed{{}}.\n\n{question}"}],
            tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = engine.tokenizer.encode(prompt, add_special_tokens=False)

        t0 = time.time()
        result = engine.generate(prompt_ids, max_new_tokens=max_tokens, temperature=0.0)
        dt = time.time() - t0

        pred = extract_boxed(result.text)
        is_correct = False
        if pred is not None:
            pred_clean = pred.replace(',', '').strip()
            is_correct = pred_clean == gold

        if is_correct:
            correct += 1
        total += 1

        status = "OK" if is_correct else "WRONG"
        tok_per_sec = len(result.token_ids) / dt
        print(f"  [{i+1}/{n_examples}] {status} | Gold={gold:>8s} Pred={str(pred):>8s} | "
              f"{len(result.token_ids)} tok in {dt:.1f}s ({tok_per_sec:.0f} tok/s) | "
              f"finish={result.finish_reason}")

        if not is_correct and i < 5:
            print(f"         Text: {result.text[:150]}...")

    elapsed = time.time() - t_start
    accuracy = correct / total * 100
    print(f"\n=== Results ===")
    print(f"Correct: {correct}/{total} = {accuracy:.1f}%")
    print(f"Total time: {elapsed:.0f}s ({elapsed/total:.1f}s/example)")


if __name__ == "__main__":
    main()
