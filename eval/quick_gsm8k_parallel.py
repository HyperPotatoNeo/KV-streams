"""Parallel GSM8K eval — multi-GPU with one model replica per GPU.

Each GPU runs an independent CompactionInferenceEngine and processes
a shard of the dataset. Results are gathered at the end.

Usage:
  torchrun --nproc_per_node=4 eval/quick_gsm8k_parallel.py [n_examples] [max_tokens]
"""

import os
import re
import sys
import time
import torch
import torch.distributed as dist


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
    n_examples = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 1024

    # Setup distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"=== Parallel GSM8K Eval ===")
        print(f"World size: {world_size}, Examples: {n_examples}, Max tokens: {max_tokens}")

    # Load model on this GPU
    sys.stdout.flush()
    from src.inference.engine import CompactionInferenceEngine

    engine = CompactionInferenceEngine(
        base_model_name="Qwen/Qwen3-0.6B-Base",
        adapter_path="outputs/ddp_scaleup_W512/condition_B/inference/adapter",
        compaction_params_path="outputs/ddp_scaleup_W512/condition_B/inference/compaction_params.pt",
        W=512, P=64, device=device, dtype_str="bfloat16",
    )

    if rank == 0:
        print(f"Model loaded on {world_size} GPUs")
        print(f"Bias mean: {engine.model.compact_attn_bias.data.mean():.3f}")

    # Load dataset on all ranks
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    examples = list(ds.select(range(n_examples)))

    # Shard examples across GPUs
    my_examples = [examples[i] for i in range(rank, n_examples, world_size)]

    if rank == 0:
        print(f"Each GPU processes ~{len(my_examples)} examples")
        print()

    dist.barrier()

    # Process my shard
    my_correct = 0
    my_total = 0
    t_start = time.time()

    for i, ex in enumerate(my_examples):
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
        is_correct = pred is not None and pred.replace(',', '').strip() == gold

        if is_correct:
            my_correct += 1
        my_total += 1

        tok_per_sec = len(result.token_ids) / dt
        status = "OK" if is_correct else "WRONG"
        global_idx = i * world_size + rank
        print(f"  [GPU{rank}] [{global_idx+1}/{n_examples}] {status} | "
              f"Gold={gold:>8s} Pred={str(pred):>8s} | "
              f"{len(result.token_ids)} tok in {dt:.1f}s ({tok_per_sec:.0f} tok/s) | "
              f"finish={result.finish_reason}", flush=True)

    elapsed = time.time() - t_start

    # Gather results across GPUs
    correct_tensor = torch.tensor([my_correct], device=device)
    total_tensor = torch.tensor([my_total], device=device)
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

    if rank == 0:
        total_correct = correct_tensor.item()
        total_count = total_tensor.item()
        accuracy = total_correct / total_count * 100
        print(f"\n=== Results ===")
        print(f"Correct: {total_correct}/{total_count} = {accuracy:.1f}%")
        print(f"Wall time: {elapsed:.0f}s ({elapsed/len(my_examples):.1f}s/example/GPU)")
        print(f"Effective throughput: {total_count/elapsed:.1f} examples/s")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
