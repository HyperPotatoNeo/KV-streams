"""Sampling utilities for token generation.

Supports temperature scaling, top-k, top-p (nucleus) sampling,
and log probability computation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = -1,
) -> tuple[int, float]:
    """Sample a single token from logits and return (token_id, log_prob).

    Args:
        logits: (vocab_size,) or (1, vocab_size) raw logits.
        temperature: Sampling temperature. 0.0 = greedy.
        top_p: Nucleus sampling threshold. 1.0 = disabled.
        top_k: Top-k filtering. -1 = disabled.

    Returns:
        (token_id, log_prob): Sampled token ID and its log probability.
    """
    if logits.dim() > 1:
        logits = logits.squeeze(0)

    # Greedy
    if temperature == 0.0 or temperature < 1e-8:
        token_id = logits.argmax(dim=-1).item()
        log_probs = F.log_softmax(logits.float(), dim=-1)
        return token_id, log_probs[token_id].item()

    # Temperature scaling
    scaled = logits.float() / temperature

    # Top-k filtering
    if top_k > 0:
        top_k = min(top_k, scaled.shape[-1])
        threshold = scaled.topk(top_k).values[-1]
        scaled = scaled.masked_fill(scaled < threshold, float("-inf"))

    # Top-p (nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = scaled.sort(descending=True)
        cumulative_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative probability above threshold
        # Shift right so that the first token above threshold is kept
        sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[sorted_mask] = float("-inf")

        # Scatter back to original indices
        scaled = scaled.scatter(0, sorted_indices, sorted_logits)

    # Sample
    probs = F.softmax(scaled, dim=-1)
    token_id = torch.multinomial(probs, num_samples=1).item()

    # Compute log probability (from original temperature-scaled logits, not filtered)
    log_probs = F.log_softmax(logits.float() / temperature, dim=-1)
    return token_id, log_probs[token_id].item()


def compute_logprob(logits: torch.Tensor, token_id: int) -> float:
    """Compute log probability of a specific token.

    Args:
        logits: (vocab_size,) or (1, vocab_size) raw logits.
        token_id: Token to compute log probability for.

    Returns:
        Log probability (float, in range (-inf, 0]).
    """
    if logits.dim() > 1:
        logits = logits.squeeze(0)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return log_probs[token_id].item()
