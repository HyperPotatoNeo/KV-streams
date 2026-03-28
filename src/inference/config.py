"""Configuration for KV Self-Compaction inference."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InferenceConfig:
    """Configuration for the compaction inference engine."""

    # Model
    base_model_name: str = "Qwen/Qwen3-0.6B-Base"
    adapter_path: Optional[str] = None
    compaction_params_path: Optional[str] = None

    # Compaction mechanism (must match training)
    W: int = 512
    P: int = 64

    # Generation defaults
    max_new_tokens: int = 2048
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 = disabled

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # Performance
    compile_forward: bool = False
    device: str = "cuda"
    dtype: str = "bfloat16"

    # Stop tokens (populated from tokenizer)
    stop_token_ids: list[int] = field(default_factory=list)
