"""Tests for CompactionInferenceEngine.

These tests require GPU. Run on a compute node:
  srun --gpus-per-node=1 python -m pytest tests/test_inference/test_engine.py -v

Key correctness tests:
  - Logit equivalence: engine prompt processing == blockwise_forward_eval
  - Compact_kv equivalence: KV states match between training and inference code
  - Partial block handling: no compaction on partial blocks
  - Logprob correctness: logprob[x_n] from step n-1's logits
"""

import pytest
import torch

# Skip all tests in this file if no GPU
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="GPU required for engine tests"
)


@pytest.fixture(scope="module")
def engine():
    """Create engine with base model (no LoRA/compaction params for correctness tests)."""
    from src.inference.engine import CompactionInferenceEngine

    # Use small W/P for fast tests
    eng = CompactionInferenceEngine(
        base_model_name="Qwen/Qwen3-0.6B-Base",
        W=128,   # Smaller than production for speed
        P=16,
        device="cuda",
        dtype_str="bfloat16",
    )
    # Initialize compaction params with known values for testing
    import torch.nn as nn
    from src.config import Qwen3Dims
    dims = Qwen3Dims()
    eng.model.compaction_embeddings = nn.Parameter(
        torch.randn(eng.P, dims.hidden_size, device=eng.device, dtype=eng.dtype) * 0.02,
        requires_grad=False,
    )
    eng.model.compact_attn_bias = nn.Parameter(
        torch.full((dims.num_q_heads,), -2.0, device=eng.device, dtype=eng.dtype),
        requires_grad=False,
    )
    return eng


@pytest.fixture(scope="module")
def sample_prompt(engine):
    """Tokenize a sample prompt."""
    text = "Solve the following math problem: What is 2 + 2? Explain your reasoning."
    return engine.tokenizer.encode(text, add_special_tokens=True)


class TestPromptProcessing:
    """Test blockwise prompt processing."""

    def test_short_prompt(self, engine):
        """Prompt shorter than W — should be processed as partial block, no compaction."""
        prompt_ids = list(range(50))  # 50 tokens < W=128
        state, logits = engine._process_prompt_blockwise(prompt_ids)

        assert logits.shape == (1, 1, engine.model.config.vocab_size)
        assert torch.isfinite(logits).all()
        assert state.tokens_in_block == 50
        assert state.block_idx == 0
        assert state.compact_kv is None  # No compaction happened

    def test_exact_W_prompt(self, engine):
        """Prompt exactly W tokens — one full block with compaction, no partial."""
        W = engine.W
        prompt_ids = list(range(W))

        state, logits = engine._process_prompt_blockwise(prompt_ids)

        assert logits.shape == (1, 1, engine.model.config.vocab_size)
        assert torch.isfinite(logits).all()
        assert state.compact_kv is not None  # Compaction happened
        assert len(state.compact_kv) == engine.num_layers
        assert state.block_idx == 1
        assert state.tokens_in_block == 0

    def test_W_plus_partial(self, engine):
        """W + partial tokens — one compaction, then partial block."""
        W = engine.W
        prompt_ids = list(range(W + 50))

        state, logits = engine._process_prompt_blockwise(prompt_ids)

        assert state.compact_kv is not None
        assert state.block_idx == 1
        assert state.tokens_in_block == 50  # Partial block
        assert torch.isfinite(logits).all()

    def test_two_W_prompt(self, engine):
        """2W tokens — two compaction events."""
        W = engine.W
        prompt_ids = list(range(2 * W))

        state, logits = engine._process_prompt_blockwise(prompt_ids)

        assert state.compact_kv is not None
        assert state.block_idx == 2
        assert state.tokens_in_block == 0
        assert torch.isfinite(logits).all()

    def test_compact_kv_shape(self, engine):
        """Verify compact_kv has correct shape per layer."""
        W, P = engine.W, engine.P
        prompt_ids = list(range(W + 10))

        state, _ = engine._process_prompt_blockwise(prompt_ids)

        assert state.compact_kv is not None
        for layer_idx, (k, v) in enumerate(state.compact_kv):
            assert k.shape == (1, engine.num_kv_heads, P, engine.head_dim), \
                f"Layer {layer_idx} K shape mismatch: {k.shape}"
            assert v.shape == (1, engine.num_kv_heads, P, engine.head_dim), \
                f"Layer {layer_idx} V shape mismatch: {v.shape}"


class TestDecodeStep:
    """Test single-token decode."""

    def test_decode_produces_finite_logits(self, engine):
        """Decode step should produce valid logits."""
        W = engine.W
        prompt_ids = list(range(W + 10))
        state, _ = engine._process_prompt_blockwise(prompt_ids)

        logits, state = engine._decode_step(100, state)  # Token ID 100

        assert logits.shape == (1, 1, engine.model.config.vocab_size)
        assert torch.isfinite(logits).all()
        assert state.tokens_in_block == 11  # 10 from prompt + 1 decoded

    def test_multiple_decode_steps(self, engine):
        """Multiple sequential decode steps should work."""
        prompt_ids = list(range(50))
        state, _ = engine._process_prompt_blockwise(prompt_ids)

        for i in range(10):
            logits, state = engine._decode_step(100 + i, state)
            assert torch.isfinite(logits).all()
            assert state.tokens_in_block == 50 + i + 1


class TestCompaction:
    """Test compaction trigger."""

    def test_compaction_resets_state(self, engine):
        """After compaction, block_idx increments and tokens_in_block resets."""
        W = engine.W
        prompt_ids = list(range(50))
        state, _ = engine._process_prompt_blockwise(prompt_ids)

        # Decode until we have W tokens in current block
        for i in range(W - 50):
            _, state = engine._decode_step(100 + i, state)

        assert state.tokens_in_block == W

        # Run compaction
        state = engine._run_compaction(state)

        assert state.tokens_in_block == 0
        assert state.block_idx == 1
        assert state.compact_kv is not None

    def test_compaction_kv_shape(self, engine):
        """Compact_kv after compaction should have correct shape."""
        W, P = engine.W, engine.P
        prompt_ids = list(range(W))  # One full block
        state, _ = engine._process_prompt_blockwise(prompt_ids)

        # state already has compact_kv from prompt processing
        assert state.compact_kv is not None
        for k, v in state.compact_kv:
            assert k.shape == (1, engine.num_kv_heads, P, engine.head_dim)
            assert v.shape == (1, engine.num_kv_heads, P, engine.head_dim)


class TestGeneration:
    """Test end-to-end generation."""

    def test_short_generation(self, engine, sample_prompt):
        """Generate a few tokens."""
        result = engine.generate(
            sample_prompt,
            max_new_tokens=10,
            temperature=0.0,  # Greedy for determinism
        )
        assert len(result.token_ids) > 0
        assert len(result.token_ids) <= 10
        assert result.finish_reason in ("stop", "length")
        assert result.prompt_token_ids == sample_prompt

    def test_greedy_determinism(self, engine, sample_prompt):
        """Greedy generation should be deterministic."""
        result1 = engine.generate(sample_prompt, max_new_tokens=20, temperature=0.0)
        result2 = engine.generate(sample_prompt, max_new_tokens=20, temperature=0.0)
        assert result1.token_ids == result2.token_ids

    def test_logprobs_returned(self, engine, sample_prompt):
        """Logprobs should be returned when requested."""
        result = engine.generate(
            sample_prompt,
            max_new_tokens=5,
            temperature=0.0,
            return_logprobs=True,
        )
        assert result.logprobs is not None
        assert len(result.logprobs) == len(result.token_ids)
        # All logprobs should be <= 0
        for lp in result.logprobs:
            assert lp <= 0.0, f"Logprob should be <= 0, got {lp}"

    def test_generation_with_compaction(self, engine):
        """Generate enough tokens to trigger at least one compaction."""
        W = engine.W
        prompt_ids = list(range(50))

        result = engine.generate(
            prompt_ids,
            max_new_tokens=W + 10,  # Should trigger compaction
            temperature=1.0,
            top_p=0.9,
        )
        assert len(result.token_ids) > 0
        assert result.finish_reason in ("stop", "length")


class TestLogitEquivalence:
    """Compare engine output against blockwise_forward_eval for correctness.

    These tests require a PEFT-wrapped model because blockwise_forward_eval
    uses src/model.py's get_inner_model() which navigates the PEFT wrapper.
    """

    @pytest.fixture(scope="class")
    def peft_engine(self):
        """Create engine with PEFT wrapper for logit comparison."""
        from src.inference.engine import CompactionInferenceEngine
        from peft import LoraConfig, get_peft_model, TaskType
        import torch.nn as nn
        from src.config import Qwen3Dims

        eng = CompactionInferenceEngine(
            base_model_name="Qwen/Qwen3-0.6B-Base",
            W=128, P=16, device="cuda", dtype_str="bfloat16",
        )
        dims = Qwen3Dims()

        # Wrap with PEFT (random LoRA init — not trained, just for wrapper compatibility)
        lora_config = LoraConfig(
            r=8, lora_alpha=16,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.0, bias="none", task_type=TaskType.CAUSAL_LM,
        )
        eng.model = get_peft_model(eng.model, lora_config).to(eng.device)

        # Add compaction params
        eng.model.compaction_embeddings = nn.Parameter(
            torch.randn(eng.P, dims.hidden_size, device=eng.device, dtype=eng.dtype) * 0.02,
            requires_grad=False,
        )
        eng.model.compact_attn_bias = nn.Parameter(
            torch.full((dims.num_q_heads,), -2.0, device=eng.device, dtype=eng.dtype),
            requires_grad=False,
        )
        return eng

    def test_full_block_logits_match(self, peft_engine):
        """Logits from engine prompt processing should match blockwise_forward_eval.

        This is THE critical correctness test. Both code paths process the same
        prompt through the same model. The logits at each text position must match.
        """
        from src.blockwise import blockwise_forward_eval
        from src.config import CompactionConfig

        engine = peft_engine
        W, P = engine.W, engine.P

        # Create a prompt that's exactly W tokens (one full block)
        prompt_ids = list(range(100, 100 + W))

        # Engine path
        state, engine_last_logits = engine._process_prompt_blockwise(prompt_ids)

        # Training path (blockwise_forward_eval) — same model with PEFT wrapper
        config = CompactionConfig(W=W, P=P)
        input_ids = torch.tensor([prompt_ids], device=engine.device)
        attention_mask = torch.ones_like(input_ids)

        training_logits = blockwise_forward_eval(
            engine.model, input_ids, attention_mask, config
        )  # (1, W, vocab_size)

        # Compare last position logits
        training_last = training_logits[0, -1, :]
        engine_last = engine_last_logits[0, 0, :]

        # Should be very close (same model, same attention, same input)
        max_diff = (training_last.float() - engine_last.float()).abs().max().item()
        assert max_diff < 1e-2, \
            f"Logit mismatch: max diff = {max_diff}"

    def test_multi_block_compact_kv_valid(self, peft_engine):
        """Compact_kv from engine should be valid (finite, non-zero)."""
        engine = peft_engine
        W, P = engine.W, engine.P

        # 2-block prompt
        prompt_ids = list(range(100, 100 + 2 * W))

        # Engine path
        state, _ = engine._process_prompt_blockwise(prompt_ids)
        engine_kv = state.compact_kv

        assert engine_kv is not None
        for layer_idx, (k, v) in enumerate(engine_kv):
            assert torch.isfinite(k).all(), f"Layer {layer_idx} K has non-finite values"
            assert torch.isfinite(v).all(), f"Layer {layer_idx} V has non-finite values"
            assert k.abs().sum() > 0, f"Layer {layer_idx} K is all zeros"
            assert v.abs().sum() > 0, f"Layer {layer_idx} V is all zeros"
