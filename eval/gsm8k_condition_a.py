"""GSM8K eval for Condition A (full context SFT, no compaction).
Uses standard HuggingFace generate(). Multi-GPU via torchrun.

Usage: torchrun --nproc_per_node=4 eval/gsm8k_condition_a.py [n_examples] [max_tokens]
"""
import os, re, sys, time
import torch
import torch.distributed as dist


def extract_boxed(text):
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    return matches[-1].strip() if matches else None

def extract_gsm8k_answer(s):
    parts = s.split('####')
    return parts[-1].strip().replace(',', '') if len(parts) > 1 else s.strip()


def main():
    n_examples = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 4096

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
    torch.cuda.set_device(device)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B-Base", dtype=torch.bfloat16,
        attn_implementation="eager", trust_remote_code=True,
    ).to(device)

    ckpt = torch.load("outputs/ddp_scaleup_W512/condition_A/checkpoint-600/checkpoint.pt",
                       map_location="cpu", weights_only=False)
    lora_config = LoraConfig(
        r=32, lora_alpha=64,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0.0, bias="none", task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    params = dict(model.named_parameters())
    for key, val in ckpt["model_state_dict"].items():
        if key in params:
            params[key].data.copy_(val.to(params[key].dtype))
    model.eval()

    if rank == 0:
        print(f"=== Condition A GSM8K (step 600) ===")
        print(f"GPUs: {world_size}, Examples: {n_examples}, Max tokens: {max_tokens}")

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    examples = list(ds.select(range(n_examples)))
    my_examples = [examples[i] for i in range(rank, n_examples, world_size)]

    dist.barrier()
    my_correct, my_total = 0, 0
    t_start = time.time()

    for i, ex in enumerate(my_examples):
        gold = extract_gsm8k_answer(ex["answer"])
        prompt = tokenizer.apply_chat_template(
            [{"role":"user","content":f"Solve the following math problem. Put the final answer in \\boxed{{}}.\n\n{ex['question']}"}],
            tokenize=False, add_generation_prompt=True,
        )
        input_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(input_ids, max_new_tokens=max_tokens, do_sample=False,
                                  pad_token_id=tokenizer.pad_token_id)
        gen = out[0][input_ids.shape[1]:].tolist()
        dt = time.time() - t0
        text = tokenizer.decode(gen, skip_special_tokens=True)
        pred = extract_boxed(text)
        ok = pred is not None and pred.replace(',','').strip() == gold
        if ok: my_correct += 1
        my_total += 1
        tps = len(gen)/dt
        print(f"  [A|GPU{rank}] [{i+1}/{len(my_examples)}] {'OK' if ok else 'WRONG'} | "
              f"Gold={gold:>8s} Pred={str(pred):>8s} | {len(gen)} tok {dt:.0f}s ({tps:.0f} t/s)",
              flush=True)

    ct = torch.tensor([my_correct], dtype=torch.int64, device=device)
    tt = torch.tensor([my_total], dtype=torch.int64, device=device)
    dist.all_reduce(ct); dist.all_reduce(tt)
    if rank == 0:
        print(f"\n=== Condition A: {ct.item()}/{tt.item()} = {ct.item()/tt.item()*100:.1f}% ===")
        print(f"Wall time: {time.time()-t_start:.0f}s")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
