"""Tests for data pipeline: chat template, label masking, padding, tokenization.

12 tests. Most use mock tokenizer for speed (no model download).
Tests that need a real Qwen3 tokenizer are marked @pytest.mark.gpu.
"""

import pytest
import torch

from src.config import CompactionConfig
from src.data import _find_assistant_token_ranges, _process_example


# ---------------------------------------------------------------------------
# Mock tokenizer for CPU tests
# ---------------------------------------------------------------------------

class MockTokenizer:
    """Minimal tokenizer mock for testing _find_assistant_token_ranges and _process_example.

    Simulates Qwen3's chat template with <|im_start|>/<|im_end|> delimiters.
    Token IDs are simple: each character is its ordinal, plus special token IDs.
    """

    # Special token IDs (chosen to not collide with ASCII range)
    IM_START_ID = 500
    IM_END_ID = 501
    EOS_ID = 502
    PAD_ID = 502  # EOS used as pad

    def __init__(self):
        self.pad_token_id = self.PAD_ID
        self.eos_token_id = self.EOS_ID
        # Precompute what "assistant\n" encodes to
        self._assistant_marker_ids = [ord(c) for c in "assistant\n"]

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<|im_start|>":
            return self.IM_START_ID
        if token == "<|im_end|>":
            return self.IM_END_ID
        raise ValueError(f"Unknown special token: {token}")

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [ord(c) for c in text]

    def apply_chat_template(self, messages: list[dict], tokenize: bool = False) -> str:
        """Produce a chat-template-formatted string like Qwen3."""
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        return "".join(parts)

    def __call__(self, text: str, add_special_tokens: bool = False,
                 return_attention_mask: bool = True) -> dict:
        """Tokenize text. Special tokens are mapped to their IDs."""
        token_ids = []
        i = 0
        while i < len(text):
            if text[i:].startswith("<|im_start|>"):
                token_ids.append(self.IM_START_ID)
                i += len("<|im_start|>")
            elif text[i:].startswith("<|im_end|>"):
                token_ids.append(self.IM_END_ID)
                i += len("<|im_end|>")
            else:
                token_ids.append(ord(text[i]))
                i += 1
        return {"input_ids": token_ids}


@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()


@pytest.fixture
def single_turn_messages():
    """Single user + assistant turn."""
    return [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there"},
    ]


@pytest.fixture
def multi_turn_messages():
    """Two user + assistant turns."""
    return [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "How?"},
        {"role": "assistant", "content": "Fine"},
    ]


@pytest.fixture
def user_only_messages():
    """User message with no assistant response."""
    return [
        {"role": "user", "content": "Hello"},
    ]


@pytest.fixture
def think_messages():
    """Assistant response with <think>...</think> tags."""
    return [
        {"role": "user", "content": "Solve 2+2"},
        {"role": "assistant", "content": "<think>Let me add 2 and 2</think>The answer is 4"},
    ]


# ---------------------------------------------------------------------------
# Tests requiring real Qwen3 tokenizer (GPU marker for model download)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_tokenizer():
    """Load real Qwen3-0.6B-Base tokenizer (requires download)."""
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base", trust_remote_code=True)


@pytest.mark.gpu
class TestRealTokenizer:
    """Tests that require the real Qwen3-0.6B-Base tokenizer."""

    def test_chat_template_has_im_tokens(self, real_tokenizer):
        """Formatted text contains <|im_start|> and <|im_end|> delimiters."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        text = real_tokenizer.apply_chat_template(messages, tokenize=False)
        assert "<|im_start|>" in text, f"Missing <|im_start|> in: {text[:200]}"
        assert "<|im_end|>" in text, f"Missing <|im_end|> in: {text[:200]}"

    def test_think_tags_in_output(self, real_tokenizer):
        """<think>...</think> tags are preserved in tokenized output."""
        messages = [
            {"role": "user", "content": "Solve 2+2"},
            {"role": "assistant", "content": "<think>Let me compute</think>4"},
        ]
        text = real_tokenizer.apply_chat_template(messages, tokenize=False)
        assert "<think>" in text, f"<think> tag missing from: {text[:200]}"
        assert "</think>" in text, f"</think> tag missing from: {text[:200]}"

        # Verify it tokenizes without error
        encoding = real_tokenizer(text, add_special_tokens=False)
        assert len(encoding["input_ids"]) > 0


# ---------------------------------------------------------------------------
# CPU tests with mock tokenizer
# ---------------------------------------------------------------------------

class TestFindAssistantRanges:
    """Tests for _find_assistant_token_ranges."""

    def test_find_assistant_ranges_single_turn(self, mock_tokenizer, single_turn_messages):
        """1 user + 1 assistant turn produces exactly 1 range."""
        text = mock_tokenizer.apply_chat_template(single_turn_messages, tokenize=False)
        encoding = mock_tokenizer(text, add_special_tokens=False)
        token_ids = encoding["input_ids"]

        ranges = _find_assistant_token_ranges(token_ids, mock_tokenizer)
        assert len(ranges) == 1, f"Expected 1 range, got {len(ranges)}: {ranges}"
        start, end = ranges[0]
        assert start < end, f"Invalid range: ({start}, {end})"

    def test_find_assistant_ranges_multi_turn(self, mock_tokenizer, multi_turn_messages):
        """2 user + 2 assistant turns produce exactly 2 ranges."""
        text = mock_tokenizer.apply_chat_template(multi_turn_messages, tokenize=False)
        encoding = mock_tokenizer(text, add_special_tokens=False)
        token_ids = encoding["input_ids"]

        ranges = _find_assistant_token_ranges(token_ids, mock_tokenizer)
        assert len(ranges) == 2, f"Expected 2 ranges, got {len(ranges)}: {ranges}"
        for start, end in ranges:
            assert start < end, f"Invalid range: ({start}, {end})"

    def test_find_assistant_ranges_no_assistant(self, mock_tokenizer, user_only_messages):
        """User-only messages produce 0 assistant ranges."""
        text = mock_tokenizer.apply_chat_template(user_only_messages, tokenize=False)
        encoding = mock_tokenizer(text, add_special_tokens=False)
        token_ids = encoding["input_ids"]

        ranges = _find_assistant_token_ranges(token_ids, mock_tokenizer)
        assert len(ranges) == 0, f"Expected 0 ranges, got {len(ranges)}: {ranges}"


class TestProcessExample:
    """Tests for _process_example: label masking, padding, truncation."""

    def test_label_masking_user_turns(self, mock_tokenizer, single_turn_messages):
        """User turn positions have labels=-100."""
        result = _process_example(single_turn_messages, mock_tokenizer, max_seq_len=512, W=16)
        assert result is not None
        labels = result["labels"]
        input_ids = result["input_ids"]

        # The user turn tokens should have labels=-100
        # Find the user content in the token stream
        text = mock_tokenizer.apply_chat_template(single_turn_messages, tokenize=False)
        encoding = mock_tokenizer(text, add_special_tokens=False)
        token_ids = encoding["input_ids"]

        # Get assistant ranges to identify non-assistant positions
        ranges = _find_assistant_token_ranges(token_ids, mock_tokenizer)
        assert len(ranges) == 1
        assistant_start, assistant_end = ranges[0]

        # All positions before the assistant content should be -100
        for i in range(min(assistant_start, len(labels))):
            assert labels[i].item() == -100, (
                f"Position {i} (before assistant) should be -100, got {labels[i].item()}"
            )

    def test_label_masking_assistant_turns(self, mock_tokenizer, single_turn_messages):
        """Assistant turn positions have actual token IDs as labels."""
        result = _process_example(single_turn_messages, mock_tokenizer, max_seq_len=512, W=16)
        assert result is not None
        labels = result["labels"]

        # At least some positions should have non-(-100) labels
        non_masked = (labels != -100).sum().item()
        assert non_masked > 0, "No assistant token labels found (all -100)"

        # Non-(-100) labels should be the NEXT token (pre-shifted for next-token prediction)
        input_ids = result["input_ids"]
        for i in range(len(labels)):
            if labels[i].item() != -100:
                assert i + 1 < len(input_ids), (
                    f"Valid label at last position {i} has no next token"
                )
                assert labels[i].item() == input_ids[i + 1].item(), (
                    f"Label at position {i}: {labels[i].item()} != next input_id {input_ids[i + 1].item()}"
                )

    def test_label_masking_padding(self, mock_tokenizer, single_turn_messages):
        """Padding positions have labels=-100."""
        result = _process_example(single_turn_messages, mock_tokenizer, max_seq_len=512, W=16)
        assert result is not None
        labels = result["labels"]
        attention_mask = result["attention_mask"]

        # Where attention_mask is 0 (padding), labels must be -100
        padding_positions = (attention_mask == 0)
        if padding_positions.any():
            padding_labels = labels[padding_positions]
            assert (padding_labels == -100).all(), (
                f"Padding positions have non-(-100) labels: {padding_labels[padding_labels != -100]}"
            )

    def test_padding_to_multiple_of_W(self, mock_tokenizer, single_turn_messages):
        """Sequence length after processing is an exact multiple of W."""
        W = 16
        result = _process_example(single_turn_messages, mock_tokenizer, max_seq_len=512, W=W)
        assert result is not None
        seq_len = result["input_ids"].shape[0]
        assert seq_len % W == 0, (
            f"seq_len={seq_len} is not a multiple of W={W}"
        )

    def test_truncation_to_max_seq_len(self, mock_tokenizer):
        """Long sequences are truncated to max_seq_len before padding."""
        # Create a very long message
        long_content = "x" * 500
        messages = [
            {"role": "user", "content": long_content},
            {"role": "assistant", "content": long_content},
        ]
        max_seq_len = 64
        W = 16
        result = _process_example(messages, mock_tokenizer, max_seq_len=max_seq_len, W=W)
        if result is not None:
            seq_len = result["input_ids"].shape[0]
            # After truncation + padding to multiple of W
            assert seq_len <= max_seq_len + W, (
                f"seq_len={seq_len} exceeds max_seq_len={max_seq_len} + W={W}"
            )
            assert seq_len % W == 0

    def test_attention_mask_matches_padding(self, mock_tokenizer, single_turn_messages):
        """attention_mask=1 where real tokens, 0 where padding."""
        result = _process_example(single_turn_messages, mock_tokenizer, max_seq_len=512, W=16)
        assert result is not None
        input_ids = result["input_ids"]
        attention_mask = result["attention_mask"]

        # Mask should be binary
        assert ((attention_mask == 0) | (attention_mask == 1)).all(), (
            "attention_mask contains non-binary values"
        )

        # Real tokens come first, padding at end
        # The sum of attention_mask equals the number of real tokens
        real_count = attention_mask.sum().item()
        assert real_count > 0, "No real tokens found"
        assert real_count <= input_ids.shape[0], "More real tokens than sequence length"

        # Check that mask transitions from 1 to 0 (no interleaving)
        if real_count < input_ids.shape[0]:
            # All 1s should come before all 0s
            assert attention_mask[:real_count].sum().item() == real_count
            assert attention_mask[real_count:].sum().item() == 0

    def test_skip_empty_examples(self, mock_tokenizer):
        """Empty or user-only messages return None (no trainable tokens)."""
        # User-only: no assistant tokens → should return None
        user_only = [{"role": "user", "content": "Hello"}]
        result = _process_example(user_only, mock_tokenizer, max_seq_len=512, W=16)
        assert result is None, "User-only messages should return None (no trainable tokens)"

    def test_skip_actually_empty(self, mock_tokenizer):
        """Truly empty messages list returns None."""
        # An empty conversation that produces no tokens
        messages = [{"role": "user", "content": ""}]
        result = _process_example(messages, mock_tokenizer, max_seq_len=512, W=16)
        # Should return None because there are no assistant tokens
        assert result is None, "Empty user content + no assistant should return None"
