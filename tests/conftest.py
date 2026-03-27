"""Shared fixtures for KV Self-Compaction Phase 2 test suite.

Provides lightweight CPU fixtures for fast unit tests.
GPU-dependent fixtures (model, tokenizer) live in their respective test files.
"""

import pytest
import torch

from src.config import CompactionConfig, Qwen3Dims


@pytest.fixture
def tiny_config() -> CompactionConfig:
    """Minimal CompactionConfig for fast CPU tests.

    W=16, P=4, K=2, max_seq_len=64 so that tensor allocations are small
    and tests complete in milliseconds.
    """
    return CompactionConfig(
        W=16,
        P=4,
        K=2,
        max_seq_len=64,
        max_steps=2,
        batch_size=1,
        eval_every=1,
        save_every=1,
    )


@pytest.fixture
def qwen3_dims() -> Qwen3Dims:
    """Real Qwen3-0.6B-Base architectural dimensions."""
    return Qwen3Dims()
