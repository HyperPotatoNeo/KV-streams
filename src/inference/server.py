"""FastAPI server with OpenAI-compatible API for KV Self-Compaction inference.

Endpoints match prime-rl's vLLM server:
  POST /v1/chat/completions          — Standard chat completions
  POST /v1/chat/completions/tokens   — Token-aware (pre-tokenized prompt)
  POST /update_weights               — Reload from checkpoint directory
  POST /load_lora_adapter            — Load PEFT adapter
  POST /tokenize                     — Tokenize messages
  GET  /health                       — Health check
  GET  /v1/models                    — Model listing

Response includes non-standard fields when return_token_ids=True:
  - prompt_token_ids (top-level)
  - token_ids (per-choice)

Usage:
  python -m src.inference.server \
      --model Qwen/Qwen3-0.6B-Base \
      --adapter outputs/condition_B/inference/adapter/ \
      --compaction-params outputs/condition_B/inference/compaction_params.pt \
      --W 512 --P 64 --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
import uuid
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from src.inference.engine import CompactionInferenceEngine, GenerationResult

logger = logging.getLogger(__name__)

# ---- Request/Response Models ----

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = ""
    messages: list[ChatMessage]
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: Optional[int] = 2048
    logprobs: bool = False
    stream: bool = False
    stop: Optional[list[str]] = None
    extra_body: Optional[dict[str, Any]] = None


class ChatCompletionRequestWithTokens(ChatCompletionRequest):
    tokens: Optional[list[int]] = None


class LogprobContent(BaseModel):
    token: str
    logprob: float


class Logprobs(BaseModel):
    content: list[LogprobContent]


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str
    logprobs: Optional[Logprobs] = None
    token_ids: Optional[list[int]] = None


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo
    prompt_token_ids: Optional[list[int]] = None


# ---- Server ----

class CompactionServer:
    """Wraps CompactionInferenceEngine in a FastAPI server."""

    def __init__(self, engine: CompactionInferenceEngine, model_name: str):
        self.engine = engine
        self.model_name = model_name
        self.gpu_lock = asyncio.Lock()
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        app = FastAPI(title="KV Self-Compaction Inference Server")

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        @app.get("/v1/models")
        async def list_models():
            return {
                "object": "list",
                "data": [{
                    "id": self.model_name,
                    "object": "model",
                    "owned_by": "local",
                }],
            }

        @app.post("/v1/chat/completions")
        async def chat_completions(request: ChatCompletionRequest):
            return await self._handle_chat(request)

        @app.post("/v1/chat/completions/tokens")
        async def chat_completions_tokens(request: ChatCompletionRequestWithTokens):
            return await self._handle_chat(request)

        @app.post("/update_weights")
        async def update_weights(request: Request):
            data = await request.json()
            weight_dir = data.get("weight_dir")
            if not weight_dir:
                raise HTTPException(400, "weight_dir required")
            async with self.gpu_lock:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, self.engine.update_weights, weight_dir
                )
            return {"status": "ok"}

        @app.post("/load_lora_adapter")
        async def load_lora_adapter(request: Request):
            data = await request.json()
            lora_name = data.get("lora_name", "")
            lora_path = data.get("lora_path", "")
            if not lora_path:
                raise HTTPException(400, "lora_path required")
            async with self.gpu_lock:
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, self.engine.load_adapter, lora_name, lora_path
                )
            return {"status": "ok"}

        @app.post("/tokenize")
        async def tokenize(request: Request):
            data = await request.json()
            messages = data.get("messages", [])
            text = self.engine.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            token_ids = self.engine.tokenizer.encode(text, add_special_tokens=False)
            return {
                "count": len(token_ids),
                "max_model_len": 4096,
                "tokens": token_ids,
            }

        return app

    async def _handle_chat(
        self, request: ChatCompletionRequest | ChatCompletionRequestWithTokens
    ) -> JSONResponse:
        """Handle chat completion request."""
        extra_body = request.extra_body or {}
        return_token_ids = extra_body.get("return_token_ids", False)

        # Tokenize prompt
        if isinstance(request, ChatCompletionRequestWithTokens) and request.tokens:
            prompt_ids = request.tokens
        else:
            text = self.engine.tokenizer.apply_chat_template(
                [m.model_dump() for m in request.messages],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_ids = self.engine.tokenizer.encode(text, add_special_tokens=False)

        # Determine stop token IDs
        stop_ids = [self.engine.tokenizer.eos_token_id]
        if request.stop:
            for s in request.stop:
                ids = self.engine.tokenizer.encode(s, add_special_tokens=False)
                if len(ids) == 1:
                    stop_ids.append(ids[0])

        # Generate (serialized GPU access)
        async with self.gpu_lock:
            loop = asyncio.get_event_loop()
            result: GenerationResult = await loop.run_in_executor(
                None,
                lambda: self.engine.generate(
                    prompt_ids=prompt_ids,
                    max_new_tokens=request.max_tokens or 2048,
                    temperature=request.temperature,
                    top_p=request.top_p,
                    top_k=extra_body.get("top_k", -1),
                    stop_token_ids=stop_ids,
                    return_logprobs=request.logprobs,
                ),
            )

        # Build response
        logprobs_obj = None
        if result.logprobs:
            logprobs_obj = Logprobs(content=[
                LogprobContent(
                    token=self.engine.tokenizer.decode([tid]),
                    logprob=lp,
                )
                for tid, lp in zip(result.token_ids, result.logprobs)
            ])

        choice = ChatCompletionChoice(
            index=0,
            message=ChatMessage(role="assistant", content=result.text),
            finish_reason=result.finish_reason,
            logprobs=logprobs_obj,
            token_ids=result.token_ids if return_token_ids else None,
        )

        response = ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex[:12]}",
            created=int(time.time()),
            model=self.model_name,
            choices=[choice],
            usage=UsageInfo(
                prompt_tokens=len(prompt_ids),
                completion_tokens=len(result.token_ids),
                total_tokens=len(prompt_ids) + len(result.token_ids),
            ),
            prompt_token_ids=prompt_ids if return_token_ids else None,
        )

        # Use model_dump with exclude_none to omit null fields
        # but keep token_ids and prompt_token_ids even when they exist
        resp_dict = response.model_dump(exclude_none=False)
        # Remove None fields except the ones we want to keep
        if not return_token_ids:
            resp_dict.pop("prompt_token_ids", None)
            if resp_dict.get("choices"):
                for c in resp_dict["choices"]:
                    c.pop("token_ids", None)

        return JSONResponse(content=resp_dict)


def main():
    parser = argparse.ArgumentParser(description="KV Self-Compaction Inference Server")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--adapter", default=None, help="PEFT adapter directory")
    parser.add_argument("--compaction-params", default=None, help="compaction_params.pt path")
    parser.add_argument("--W", type=int, default=512)
    parser.add_argument("--P", type=int, default=64)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    engine = CompactionInferenceEngine(
        base_model_name=args.model,
        adapter_path=args.adapter,
        compaction_params_path=args.compaction_params,
        W=args.W,
        P=args.P,
        device=args.device,
        dtype_str=args.dtype,
    )
    print("Model loaded.")

    server = CompactionServer(engine, model_name=args.model)
    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(server.app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
