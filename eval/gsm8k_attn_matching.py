"""GSM8K eval with attention matching compaction on full-context model.

Takes the Condition A (full context SFT) model and applies post-hoc
KV cache compression via attention matching with random query probes.

Algorithm:
  1. Generate tokens normally using the full-context model
  2. Every W tokens, compress the KV cache:
     a. Sample random query probes
     b. Compute attention scores between probes and all KV entries
     c. Score each KV by sqrt(variance) of attention across probes
     d. Keep top-P entries, discard the rest
  3. Continue generation with compressed cache

Usage:
  torchrun --nproc_per_node=4 eval/gsm8k_attn_matching.py [--n 50] [--max-tokens 4096]
"""

import argparse, math, os, re, sys, time
import torch
import torch.nn.functional as F
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


def get_cache_kv(cache, layer_idx):
    """Get (K, V) from cache, v5-compatible."""
    if hasattr(cache, 'layers') and len(cache.layers) > layer_idx:
        return cache.layers[layer_idx].keys, cache.layers[layer_idx].values
    return cache[layer_idx]


def attention_matching_compress(cache, num_layers, target_len, num_probes=64, device='cuda'):
    """Compress KV cache to target_len entries using attention matching.

    Uses random query probes to score KV importance via sqrt(variance) of attention.
    Selects top-target_len entries across all positions.

    Args:
        cache: DynamicCache with current KV entries
        num_layers: Number of decoder layers
        target_len: Number of KV entries to keep (P)
        num_probes: Number of random query probes
        device: torch device

    Returns:
        New DynamicCache with only top-target_len entries per layer
    """
    # Get dimensions from first layer
    k0, v0 = get_cache_kv(cache, 0)
    B, num_kv_heads, seq_len, head_dim = k0.shape

    if seq_len <= target_len:
        return cache  # Nothing to compress

    # Generate random query probes: (num_probes, num_kv_heads, head_dim)
    probes = torch.randn(num_probes, num_kv_heads, head_dim,
                         device=device, dtype=k0.dtype)
    probes = F.normalize(probes, dim=-1)  # Unit norm for stable attention

    # Compute importance scores across all layers
    # Score = mean across layers of sqrt(var across probes of attention weights)
    all_scores = torch.zeros(seq_len, device=device, dtype=torch.float32)

    scale = 1.0 / math.sqrt(head_dim)

    for layer_idx in range(num_layers):
        k, v = get_cache_kv(cache, layer_idx)
        # k shape: (B, num_kv_heads, seq_len, head_dim)
        k_squeezed = k[0]  # (num_kv_heads, seq_len, head_dim) — batch=1

        # Compute attention: probes @ k^T
        # probes: (num_probes, num_kv_heads, head_dim)
        # k_squeezed: (num_kv_heads, seq_len, head_dim)
        # scores: (num_probes, num_kv_heads, seq_len)
        attn_logits = torch.einsum('phd,hsd->phs', probes, k_squeezed) * scale
        attn_weights = F.softmax(attn_logits.float(), dim=-1)  # (num_probes, num_kv_heads, seq_len)

        # Importance = sqrt(variance across probes), averaged across heads
        # Higher variance = more "distinctive" KV entries
        var_across_probes = attn_weights.var(dim=0)  # (num_kv_heads, seq_len)
        importance = var_across_probes.sqrt().mean(dim=0)  # (seq_len,)
        all_scores += importance

    # Average across layers
    all_scores /= num_layers

    # Select top-target_len indices
    _, top_indices = all_scores.topk(target_len)
    top_indices = top_indices.sort().values  # Keep temporal order

    # Build new cache with selected entries
    new_cache = DynamicCache()
    for layer_idx in range(num_layers):
        k, v = get_cache_kv(cache, layer_idx)
        # Select entries: k[:, :, top_indices, :]
        k_selected = k[:, :, top_indices, :].clone()
        v_selected = v[:, :, top_indices, :].clone()
        new_cache.update(k_selected, v_selected, layer_idx)

    return new_cache


def generate_with_attn_matching(model, tokenizer, prompt_ids, max_new_tokens, W, P, device):
    """Generate text with periodic attention matching compression.

    Uses standard HF generate for token-by-token decode, but compresses
    the KV cache every W tokens using attention matching.
    """
    num_layers = model.config.num_hidden_layers
    input_ids = torch.tensor([prompt_ids], device=device)
    generated = []

    # Initial forward to fill cache
    with torch.inference_mode():
        outputs = model(input_ids, use_cache=True)
        cache = outputs.past_key_values
        logits = outputs.logits[:, -1, :]  # Last position

    tokens_since_compress = len(prompt_ids)

    # Check if we need initial compression
    if tokens_since_compress >= W:
        # Compress multiple times if prompt > W
        while tokens_since_compress >= W:
            cache = attention_matching_compress(cache, num_layers, P, device=device)
            tokens_since_compress = P  # After compression, cache has P entries

    for step in range(max_new_tokens):
        # Sample greedily
        token_id = logits.argmax(dim=-1).item()
        generated.append(token_id)

        if token_id == tokenizer.eos_token_id:
            break

        # Forward one token
        token_tensor = torch.tensor([[token_id]], device=device)
        with torch.inference_mode():
            outputs = model(token_tensor, past_key_values=cache, use_cache=True)
            cache = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

        tokens_since_compress += 1

        # Compression trigger
        if tokens_since_compress >= W:
            cache = attention_matching_compress(cache, num_layers, P, device=device)
            tokens_since_compress = P  # Cache now has P entries

    text = tokenizer.decode(generated, skip_special_tokens=True)
    return generated, text


def main():
    parser = argparse.ArgumentParser()
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

    # Load Condition A model (full context SFT)
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B-Base', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen3-0.6B-Base', dtype=torch.bfloat16,
        attn_implementation='eager', trust_remote_code=True,
    ).to(device)

    # Load Condition A LoRA
    ckpt = torch.load('outputs/ddp_scaleup_W512/condition_A/checkpoint-600/checkpoint.pt',
                       map_location='cpu', weights_only=False)
    lora_config = LoraConfig(
        r=32, lora_alpha=64,
        target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
        lora_dropout=0.0, bias='none', task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    params = dict(model.named_parameters())
    for key, val in ckpt['model_state_dict'].items():
        if key in params:
            params[key].data.copy_(val.to(params[key].dtype))
    model.eval()

    if rank == 0:
        print(f'=== Attn Matching Compaction on Condition A ===')
        print(f'W={args.W} P={args.P} max_tokens={args.max_tokens} n={args.n}')

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
        gen_ids, text = generate_with_attn_matching(
            model, tokenizer, prompt_ids, args.max_tokens, args.W, args.P, device
        )
        dt = time.time() - t0

        pred = extract_boxed(text)
        ok = pred is not None and pred.replace(',', '').strip() == gold
        if ok: my_correct += 1
        tps = len(gen_ids) / dt if dt > 0 else 0
        status = 'OK' if ok else 'WRONG'
        print(f'  [AM|GPU{rank}] [{i+1}/{len(my_examples)}] {status} | '
              f'Gold={gold:>8s} Pred={str(pred):>8s} | '
              f'{len(gen_ids)} tok {dt:.0f}s ({tps:.0f} t/s)', flush=True)

    ct = torch.tensor([my_correct], dtype=torch.int64, device=device)
    tt = torch.tensor([len(my_examples)], dtype=torch.int64, device=device)
    dist.all_reduce(ct); dist.all_reduce(tt)
    if rank == 0:
        print(f'\n=== Attn Matching: {ct.item()}/{tt.item()} = {ct.item()/tt.item()*100:.1f}% ===')
        print(f'Wall time: {time.time()-t_start:.0f}s')
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
