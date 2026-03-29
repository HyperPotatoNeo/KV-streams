"""GSM8K eval for Condition E (blockwise, no compact_kv between blocks).

Condition E processes in W-sized blocks but does NOT pass compact_kv between them.
Each block starts fresh — effectively truncation to the last W tokens of context.
The model still generates compaction tokens within each block but they are discarded.

For inference: we just use standard generation but clear the KV cache every W tokens,
keeping nothing from previous blocks. This is equivalent to a W-token sliding window
with no overlap.

Usage:
  torchrun --nproc_per_node=4 eval/gsm8k_condition_e.py [--n 50] [--max-tokens 4096]
"""

import argparse, os, re, sys, time
import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from peft import LoraConfig, get_peft_model, TaskType


def extract_boxed(text):
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    return matches[-1].strip() if matches else None

def extract_gsm8k_answer(s):
    parts = s.split('####')
    return parts[-1].strip().replace(',', '') if len(parts) > 1 else s.strip()


def generate_condition_e(model, tokenizer, prompt_ids, max_new_tokens, W, device):
    """Generate with Condition E: clear KV cache every W tokens (no cross-block state)."""
    input_ids = torch.tensor([prompt_ids], device=device)
    generated = []

    # Initial forward to fill cache
    with torch.inference_mode():
        outputs = model(input_ids, use_cache=True)
        cache = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

    tokens_in_block = len(prompt_ids)

    # If prompt > W, clear cache and keep only last W tokens' KV
    if tokens_in_block > W:
        # Re-process just the last W tokens
        last_w = input_ids[:, -W:]
        with torch.inference_mode():
            outputs = model(last_w, use_cache=True)
            cache = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
        tokens_in_block = W

    for step in range(max_new_tokens):
        token_id = logits.argmax(dim=-1).item()
        generated.append(token_id)

        if token_id == tokenizer.eos_token_id:
            break

        token_tensor = torch.tensor([[token_id]], device=device)
        with torch.inference_mode():
            outputs = model(token_tensor, past_key_values=cache, use_cache=True)
            cache = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

        tokens_in_block += 1

        # Clear cache every W tokens (Condition E: no cross-block state)
        if tokens_in_block >= W:
            # Discard all context, keep nothing
            # Re-process the last generated token with empty cache
            cache = DynamicCache()
            with torch.inference_mode():
                outputs = model(token_tensor, use_cache=True)
                cache = outputs.past_key_values
                logits = outputs.logits[:, -1, :]
            tokens_in_block = 1

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return generated, text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--W', type=int, default=512)
    args = parser.parse_args()

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f'cuda:{int(os.environ.get("LOCAL_RANK", 0))}'
    torch.cuda.set_device(device)

    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16,
        attn_implementation='eager', trust_remote_code=True,
    ).to(device)

    ckpt = torch.load('outputs/ddp_scaleup_W512_E/condition_E/checkpoint-600/checkpoint.pt',
                       map_location='cpu', weights_only=False)
    lora_config = LoraConfig(
        r=32, lora_alpha=64,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        lora_dropout=0.0, bias='none', task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    params = dict(model.named_parameters())
    loaded = 0
    for key, val in ckpt['model_state_dict'].items():
        if key in params:
            params[key].data.copy_(val.to(params[key].dtype))
            loaded += 1
    model.eval()

    if rank == 0:
        print(f'=== Condition E GSM8K (step 600) ===')
        print(f'W={args.W} max_tokens={args.max_tokens} n={args.n}')
        print(f'Loaded {loaded} params')

    from datasets import load_dataset
    ds = load_dataset('openai/gsm8k', 'main', split='test')
    examples = list(ds.select(range(args.n)))
    my_examples = [examples[i] for i in range(rank, args.n, world_size)]

    dist.barrier()
    my_correct = 0
    t_start = time.time()

    for i, ex in enumerate(my_examples):
        gold = extract_gsm8k_answer(ex['answer'])
        prompt = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': f'Solve the following math problem. Put the final answer in \\boxed{{}}.\n\n{ex["question"]}'}],
            tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        t0 = time.time()
        gen_ids, text = generate_condition_e(model, tokenizer, prompt_ids, args.max_tokens, args.W, device)
        dt = time.time() - t0
        pred = extract_boxed(text)
        ok = pred is not None and pred.replace(',', '').strip() == gold
        if ok: my_correct += 1
        tps = len(gen_ids) / dt if dt > 0 else 0
        status = 'OK' if ok else 'WRONG'
        print(f'  [E|GPU{rank}] [{i+1}/{len(my_examples)}] {status} | '
              f'Gold={gold:>8s} Pred={str(pred):>8s} | '
              f'{len(gen_ids)} tok {dt:.0f}s ({tps:.0f} t/s)', flush=True)

    ct = torch.tensor([my_correct], dtype=torch.int64, device=device)
    tt = torch.tensor([len(my_examples)], dtype=torch.int64, device=device)
    dist.all_reduce(ct); dist.all_reduce(tt)
    if rank == 0:
        print(f'\n=== Condition E: {ct.item()}/{tt.item()} = {ct.item()/tt.item()*100:.1f}% ===')
        print(f'Wall time: {time.time()-t_start:.0f}s')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
