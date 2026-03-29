"""Tests for the FastAPI server API compatibility with prime-rl.

Tests run against a test client (no GPU needed for API format tests).
"""

import pytest
import json
from unittest.mock import MagicMock, AsyncMock, patch

from src.inference.server import CompactionServer, ChatCompletionRequest, ChatMessage
from src.inference.engine import GenerationResult


@pytest.fixture
def mock_engine():
    """Create a mock engine that returns predictable results."""
    engine = MagicMock()
    engine.tokenizer = MagicMock()
    engine.tokenizer.eos_token_id = 151643
    engine.tokenizer.bos_token_id = 151643
    engine.tokenizer.encode.return_value = [100, 200, 300]
    engine.tokenizer.decode.return_value = "test output"
    engine.tokenizer.apply_chat_template.return_value = "formatted text"

    engine.generate.return_value = GenerationResult(
        token_ids=[400, 500, 600],
        text="test output",
        logprobs=[-0.5, -1.0, -0.3],
        finish_reason="stop",
        prompt_token_ids=[100, 200, 300],
    )
    return engine


@pytest.fixture
def server(mock_engine):
    return CompactionServer(mock_engine, model_name="test-model")


@pytest.fixture
def client(server):
    from fastapi.testclient import TestClient
    return TestClient(server.app)


class TestHealthEndpoint:

    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestModelsEndpoint:

    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1
        assert data["data"][0]["id"] == "test-model"


class TestChatCompletions:

    def test_basic_completion(self, client, mock_engine):
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert data["model"] == "test-model"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["finish_reason"] == "stop"
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "test output"

    def test_return_token_ids(self, client, mock_engine):
        """Critical: prime-rl expects prompt_token_ids and token_ids in response."""
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "extra_body": {"return_token_ids": True},
        })
        assert resp.status_code == 200
        data = resp.json()

        # Non-standard fields that prime-rl/verifiers expect
        assert "prompt_token_ids" in data, "Missing prompt_token_ids at top level"
        assert data["prompt_token_ids"] == [100, 200, 300]

        assert "token_ids" in data["choices"][0], "Missing token_ids in choice"
        assert data["choices"][0]["token_ids"] == [400, 500, 600]

    def test_no_token_ids_by_default(self, client, mock_engine):
        """When return_token_ids not set, response should NOT have token fields."""
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        data = resp.json()
        assert "prompt_token_ids" not in data or data["prompt_token_ids"] is None
        assert "token_ids" not in data["choices"][0] or data["choices"][0]["token_ids"] is None

    def test_logprobs_format(self, client, mock_engine):
        """Logprobs must be in OpenAI format: logprobs.content[i].logprob"""
        mock_engine.generate.return_value = GenerationResult(
            token_ids=[400, 500],
            text="hi",
            logprobs=[-0.5, -1.0],
            finish_reason="stop",
            prompt_token_ids=[100],
        )
        mock_engine.tokenizer.decode.side_effect = lambda ids: "t" + str(ids[0])

        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "logprobs": True,
        })
        data = resp.json()
        logprobs = data["choices"][0]["logprobs"]
        assert logprobs is not None
        assert "content" in logprobs
        assert len(logprobs["content"]) == 2
        assert "logprob" in logprobs["content"][0]
        assert logprobs["content"][0]["logprob"] == -0.5
        assert logprobs["content"][1]["logprob"] == -1.0

    def test_usage_info(self, client, mock_engine):
        resp = client.post("/v1/chat/completions", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        data = resp.json()
        assert "usage" in data
        assert data["usage"]["prompt_tokens"] == 3  # [100, 200, 300]
        assert data["usage"]["completion_tokens"] == 3  # [400, 500, 600]
        assert data["usage"]["total_tokens"] == 6


class TestTokenEndpoint:

    def test_chat_with_tokens(self, client, mock_engine):
        """Token-aware endpoint should use provided tokens instead of tokenizing."""
        resp = client.post("/v1/chat/completions/tokens", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "tokens": [1, 2, 3, 4, 5],
        })
        assert resp.status_code == 200

        # Verify the engine was called with the provided tokens
        call_args = mock_engine.generate.call_args
        assert call_args.kwargs.get("prompt_ids") == [1, 2, 3, 4, 5] or \
               call_args[1].get("prompt_ids") == [1, 2, 3, 4, 5] or \
               (len(call_args[0]) > 0 and call_args[0][0] == [1, 2, 3, 4, 5])


class TestWeightUpdate:

    def test_update_weights_schema(self, client, mock_engine):
        """Weight update must accept {weight_dir: ...} and return {status: ok}."""
        resp = client.post("/update_weights", json={"weight_dir": "/path/to/weights"})
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_update_weights_missing_field(self, client):
        """Should return 400 if weight_dir is missing."""
        resp = client.post("/update_weights", json={})
        assert resp.status_code == 400


class TestTokenizeEndpoint:

    def test_tokenize(self, client, mock_engine):
        mock_engine.tokenizer.apply_chat_template.return_value = "formatted"
        mock_engine.tokenizer.encode.return_value = [10, 20, 30, 40]

        resp = client.post("/tokenize", json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "count" in data
        assert "tokens" in data
        assert data["count"] == 4
        assert data["tokens"] == [10, 20, 30, 40]
