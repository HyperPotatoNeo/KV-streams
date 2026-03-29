"""GSM8K eval for both Condition A (full context) and B (compaction).

Condition A: Standard LoRA SFT, no compaction — uses HuggingFace generate()
Condition B: KV Self-Compaction — uses CompactionInferenceEngine

Runs both on separate GPUs in parallel via torchrun.
GPU 0,1 = Condition A (2 replicas), GPU 2,3 = Condition B (2 replicas)

Usage:
  torchrun --nproc_per_node=4 --master_port=29999 eval/gsm8k_both_conditions.py [n_examples] [max_tokens]
"""

import os
import re
import sys
import time
import torch
import torch.distributed as dist


def extract_boxed(text):
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    return matches[-1].strip() if matches else None


def extract_gsm8k_answer(answer_str):
    parts = answer_str.split('####')
    return parts[-1].strip().replace(',', '') if len(parts) > 1 else answer_str.strip()


def run_condition_a(examples, max_tokens, device, rank, world_size):
    """Condition A: standard HuggingFace generate (no compaction)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B-Base", dtype=torch.bfloat16,
        attn_implementation="eager", trust_remote_code=True,
    ).to(device)

    # Convert and load Condition A checkpoint
    ckpt_path = "outputs/ddp_scaleup_W512/condition_A/checkpoint-600/checkpoint.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]

    # Condition A has LoRA params but NO compaction params
    from peft import LoraConfig, get_peft_model, TaskType
    lora_config = LoraConfig(
        r=32, lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # Load weights
    model_params = dict(model.named_parameters())
    for key, value in state_dict.items():
        if key in model_params:
            model_params[key].data.copy_(value.to(model_params[key].dtype))
    model.eval()

    correct = 0
    total = 0

    for i, ex in enumerate(examples):
        question = ex["question"]
        gold = extract_gsm8k_answer(ex["answer"])

        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": f"Solve the following math problem. Put the final answer in \\boxed{{}}.\n\n{question}"}],
            tokenize=False, add_generation_prompt=True,
        )
        input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)

        t0 = time.time()
        with torch.inference_mode():
            output = model.generate(
                input_ids, max_new_tokens=max_tokens, temperature=1.0,
                do_sample=False, pad_token_id=tokenizer.pad_token_id,
            )
        gen_ids = output[0][input_ids.shape[1]:].tolist()
        dt = time.time() - t0

        text = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred = extract_boxed(text)
        is_correct = pred is not None and pred.replace(',', '').strip() == gold

        if is_correct:
            correct += 1
        total += 1

        tok_per_sec = len(gen_ids) / dt
        status = "OK" if is_correct else "WRONG"
        print(f"  [A|GPU{rank}] [{i+1}/{len(examples)}] {status} | "
              f"Gold={gold:>8s} Pred={str(pred):>8s} | "
              f"{len(gen_ids)} tok in {dt:.1f}s ({tok_per_sec:.0f} tok/s) | fin={output[0][-1].item() == tokenizer.eos_token_id}",
              flush=True)

    return correct, total


def run_condition_b(examples, max_tokens, device, rank, world_size):
    """Condition B: KV Self-Compaction inference."""
    from src.inference.engine import CompactionInferenceEngine

    # Convert checkpoint if not already done
    inference_dir = "outputs/ddp_scaleup_W512/condition_B/inference/"
    if not os.path.exists(os.path.join(inference_dir, "compaction_params.pt")):
        if rank == 2:  # Only one rank converts
            from src.inference.convert_checkpoint import convert_training_checkpoint
            convert_training_checkpoint(
                "outputs/ddp_scaleup_W512/condition_B/checkpoint-600/checkpoint.pt",
                "Qwen/Qwen3-0.6B-Base", inference_dir,
            )

    dist.barrier()

    engine = CompactionInferenceEngine(
        base_model_name="Qwen/Qwen3-0.6B-Base",
        adapter_path=os.path.join(inference_dir, "adapter"),
        compaction_params_path=os.path.join(inference_dir, "compaction_params.pt"),
        W=512, P=64, device=device, dtype_str="bfloat16",
    )

    correct = 0
    total = 0

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
        is_correct = pred is not None and pred.replace(',', '').strip() == gold

        if is_correct:
            correct += 1
        total += 1

        tok_per_sec = len(result.token_ids) / dt
        status = "OK" if is_correct else "WRONG"
        print(f"  [B|GPU{rank}] [{i+1}/{len(examples)}] {status} | "
              f"Gold={gold:>8s} Pred={str(pred):>8s} | "
              f"{len(result.token_ids)} tok in {dt:.1f}s ({tok_per_sec:.0f} tok/s) | fin={result.finish_reason}",
              flush=True)

    return correct, total


def main():
    n_examples = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 4096

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}"
    torch.cuda.set_device(device)

    if rank == 0:
        print(f"=== GSM8K: Condition A vs B ===")
        print(f"World size: {world_size}, Examples: {n_examples}, Max tokens: {max_tokens}")
        print(f"GPU 0,1 = Condition A | GPU 2,3 = Condition B")

    # Load dataset on all ranks
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    examples = list(ds.select(range(n_examples)))

    dist.barrier()

    # Split: ranks 0,1 do Condition A, ranks 2,3 do Condition B
    if rank < 2:
        # Condition A: shard among ranks 0,1
        condition_rank = rank  # 0 or 1
        condition_size = 2
        my_examples = [examples[i] for i in range(condition_rank, n_examples, condition_size)]
        my_correct, my_total = run_condition_a(my_examples, max_tokens, device, rank, world_size)
    else:
        # Condition B: shard among ranks 2,3
        condition_rank = rank - 2  # 0 or 1
        condition_size = 2
        my_examples = [examples[i] for i in range(condition_rank, n_examples, condition_size)]
        my_correct, my_total = run_condition_b(my_examples, max_tokens, device, rank, world_size)

    # Gather results to rank 0 via all_gather
    correct_t = torch.tensor([my_correct], dtype=torch.int64, device=device)
    total_t = torch.tensor([my_total], dtype=torch.int64, device=device)

    all_correct = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)]
    all_total = [torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)]
    dist.all_gather(all_correct, correct_t)
    dist.all_gather(all_total, total_t)

    if rank == 0:
        a_correct = sum(all_correct[i].item() for i in range(2))
        a_total = sum(all_total[i].item() for i in range(2))
        b_correct = sum(all_correct[i].item() for i in range(2, 4))
        b_total = sum(all_total[i].item() for i in range(2, 4))

        print(f"\n{'='*50}")
        print(f"=== RESULTS ===")
        print(f"Condition A (full context): {int(a_correct)}/{int(a_total)} = {a_correct/a_total*100:.1f}%")
        print(f"Condition B (compaction):   {int(b_correct)}/{int(b_total)} = {b_correct/b_total*100:.1f}%")
        print(f"{'='*50}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
