"""Convert training checkpoint to inference format.

Training saves a flat checkpoint.pt with:
  - model_state_dict: LoRA weights + compaction params (PEFT naming)
  - optimizer_state_dict, scheduler_state_dict, step, config

This script produces:
  - output_dir/adapter/  — PEFT adapter (adapter_model.safetensors + adapter_config.json)
  - output_dir/compaction_params.pt  — {compaction_embeddings, compact_attn_bias}

Usage:
  python -m src.inference.convert_checkpoint \
      --ckpt outputs/condition_B/checkpoint_step600.pt \
      --base-model Qwen/Qwen3-0.6B-Base \
      --output outputs/condition_B/inference/
"""

from __future__ import annotations

import argparse
import os

import torch


def convert_training_checkpoint(
    ckpt_path: str,
    base_model_name: str,
    output_dir: str,
    lora_rank: int = 32,
    lora_alpha: int = 64,
    lora_targets: list[str] | None = None,
) -> None:
    """Convert training checkpoint to inference format.

    Args:
        ckpt_path: Path to training checkpoint.pt
        base_model_name: HuggingFace model ID for base model
        output_dir: Directory to write adapter/ and compaction_params.pt
        lora_rank: LoRA rank (must match training)
        lora_alpha: LoRA alpha (must match training)
        lora_targets: LoRA target modules (must match training)
    """
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, TaskType

    if lora_targets is None:
        lora_targets = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"]

    # Separate compaction params from LoRA params
    compaction_keys = {"compaction_embeddings", "compact_attn_bias"}
    compaction_state = {}
    lora_state = {}

    for key, value in state_dict.items():
        if key in compaction_keys:
            compaction_state[key] = value
        else:
            lora_state[key] = value

    print(f"Found {len(lora_state)} LoRA params, {len(compaction_state)} compaction params")

    # Save compaction params
    os.makedirs(output_dir, exist_ok=True)
    compaction_path = os.path.join(output_dir, "compaction_params.pt")
    torch.save(compaction_state, compaction_path)
    print(f"Saved compaction params to {compaction_path}")

    # Reconstruct PeftModel and load LoRA weights
    print(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=lora_targets,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(base_model, lora_config)

    # Load LoRA weights into the model
    missing, unexpected = [], []
    model_state = dict(peft_model.named_parameters())
    for key, value in lora_state.items():
        if key in model_state:
            model_state[key].data.copy_(value)
        else:
            unexpected.append(key)

    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys: {unexpected[:5]}...")

    # Save as PEFT adapter
    adapter_dir = os.path.join(output_dir, "adapter")
    peft_model.save_pretrained(adapter_dir)
    print(f"Saved PEFT adapter to {adapter_dir}")

    # Verify roundtrip with a fresh base model (original is mutated by get_peft_model)
    print("Verifying roundtrip...")
    from peft import PeftModel
    fresh_base = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.bfloat16,
        attn_implementation="eager", trust_remote_code=True,
    )
    reloaded = PeftModel.from_pretrained(fresh_base, adapter_dir)
    mismatches = 0
    for key, value in lora_state.items():
        if key in dict(reloaded.named_parameters()):
            reloaded_val = dict(reloaded.named_parameters())[key].data
            if not torch.equal(value.to(reloaded_val.dtype), reloaded_val):
                print(f"  MISMATCH: {key}")
                mismatches += 1
    if mismatches == 0:
        print("  All LoRA weights match!")
    print("Roundtrip verification complete")


def main():
    parser = argparse.ArgumentParser(description="Convert training checkpoint to inference format")
    parser.add_argument("--ckpt", required=True, help="Path to training checkpoint.pt")
    parser.add_argument("--base-model", default="Qwen/Qwen3-0.6B-Base", help="Base model name")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    args = parser.parse_args()

    convert_training_checkpoint(
        ckpt_path=args.ckpt,
        base_model_name=args.base_model,
        output_dir=args.output,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )


if __name__ == "__main__":
    main()
