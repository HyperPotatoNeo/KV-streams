"""Generic GSM8K eval for any compaction checkpoint.
Multi-GPU via torchrun.

Usage:
  torchrun --nproc_per_node=4 eval/gsm8k_generic.py \
      --adapter PATH --compaction-params PATH [--n 50] [--max-tokens 4096]
"""
import argparse, os, re, sys, time, torch
import torch.distributed as dist


def extract_boxed(text):
    matches = re.findall(r'\\boxed\{([^}]*)\}', text)
    return matches[-1].strip() if matches else None

def extract_gsm8k_answer(s):
    parts = s.split('####')
    return parts[-1].strip().replace(',', '') if len(parts) > 1 else s.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--compaction-params', required=True)
    parser.add_argument('--label', default='eval')
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--W', type=int, default=512)
    parser.add_argument('--P', type=int, default=64)
    args = parser.parse_args()

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f'cuda:{int(os.environ.get("LOCAL_RANK", 0))}'
    torch.cuda.set_device(device)

    from src.inference.engine import CompactionInferenceEngine
    engine = CompactionInferenceEngine(
        base_model_name='Qwen/Qwen3-0.6B-Base',
        adapter_path=args.adapter,
        compaction_params_path=args.compaction_params,
        W=args.W, P=args.P, device=device, dtype_str='bfloat16',
    )
    if rank == 0:
        print(f'=== {args.label} GSM8K ===')
        print(f'W={args.W} P={args.P} max_tokens={args.max_tokens} n={args.n}')
        print(f'Bias: {engine.model.compact_attn_bias.data.mean():.3f}')

    from datasets import load_dataset
    ds = load_dataset('openai/gsm8k', 'main', split='test')
    examples = list(ds.select(range(args.n)))
    my_examples = [examples[i] for i in range(rank, args.n, world_size)]

    dist.barrier()
    my_correct = 0
    t_start = time.time()

    for i, ex in enumerate(my_examples):
        gold = extract_gsm8k_answer(ex['answer'])
        prompt = engine.tokenizer.apply_chat_template(
            [{'role': 'user', 'content': f'Solve the following math problem. Put the final answer in \\boxed{{}}.\n\n{ex["question"]}'}],
            tokenize=False, add_generation_prompt=True,
        )
        prompt_ids = engine.tokenizer.encode(prompt, add_special_tokens=False)
        t0 = time.time()
        result = engine.generate(prompt_ids, max_new_tokens=args.max_tokens, temperature=0.0)
        dt = time.time() - t0
        pred = extract_boxed(result.text)
        ok = pred is not None and pred.replace(',', '').strip() == gold
        if ok: my_correct += 1
        tps = len(result.token_ids) / dt
        status = 'OK' if ok else 'WRONG'
        print(f'  [{args.label}|GPU{rank}] [{i+1}/{len(my_examples)}] {status} | '
              f'Gold={gold:>8s} Pred={str(pred):>8s} | '
              f'{len(result.token_ids)} tok {dt:.0f}s ({tps:.0f} t/s)', flush=True)

    ct = torch.tensor([my_correct], dtype=torch.int64, device=device)
    tt = torch.tensor([len(my_examples)], dtype=torch.int64, device=device)
    dist.all_reduce(ct); dist.all_reduce(tt)
    if rank == 0:
        print(f'\n=== {args.label}: {ct.item()}/{tt.item()} = {ct.item()/tt.item()*100:.1f}% ===')
        print(f'Wall time: {time.time()-t_start:.0f}s')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
