"""Model pool: streaming calls to a small and a large model, with timeout, fallback and cost accounting."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Protocol

from llm_router.config import ModelSpec

TIERS = ("small", "large")


@dataclass
class Chunk:
    text: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None  # includes thinking tokens, which are billed as output


class Backend(Protocol):
    def stream(self, prompt: str) -> AsyncIterator[Chunk]: ...


@dataclass
class Result:
    text: str
    tier: str
    model: str
    ttft_s: float
    total_s: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    fell_back: bool = False
    cached: bool = False

    def meta(self) -> dict:
        return {k: v for k, v in self.__dict__.items() if k != "text"}


class GeminiBackend:
    def __init__(self, client, model: str):
        self._client = client  # keep a reference so the SDK's HTTP client is not collected mid-call
        self._model = model

    async def stream(self, prompt: str) -> AsyncIterator[Chunk]:
        stream = await self._client.aio.models.generate_content_stream(model=self._model, contents=prompt)
        async for chunk in stream:
            usage = chunk.usage_metadata
            inp = out = None
            if usage is not None:
                inp = usage.prompt_token_count
                out = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
            yield Chunk(text=chunk.text or "", input_tokens=inp, output_tokens=out)


class PreTokenError(Exception):
    """A backend failed or timed out before producing any text, so falling back is safe."""


class Pool:
    def __init__(self, backends: dict[str, Backend], specs: dict[str, ModelSpec], timeout_s: float = 30.0):
        self.backends, self.specs, self.timeout_s = backends, specs, timeout_s

    async def _run(self, prompt: str, tier: str, fell_back: bool) -> AsyncIterator[tuple[str, object]]:
        t0 = time.perf_counter()
        it = self.backends[tier].stream(prompt).__aiter__()
        parts: list[str] = []
        ttft: float | None = None
        inp = out = 0
        first = True
        while True:
            try:
                chunk = await asyncio.wait_for(it.__anext__(), self.timeout_s)
            except StopAsyncIteration:
                if first:
                    raise PreTokenError(f"{tier} returned an empty stream") from None
                break
            except Exception as exc:
                if first:
                    raise PreTokenError(f"{tier}: {type(exc).__name__}: {exc}") from exc
                raise
            first = False
            if chunk.input_tokens is not None:
                inp = chunk.input_tokens
            if chunk.output_tokens is not None:
                out = chunk.output_tokens
            if chunk.text:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                parts.append(chunk.text)
                yield "delta", chunk.text
        total = time.perf_counter() - t0
        spec = self.specs[tier]
        yield "done", Result(
            text="".join(parts), tier=tier, model=spec.name, ttft_s=ttft if ttft is not None else total,
            total_s=total, input_tokens=inp, output_tokens=out, cost_usd=spec.cost(inp, out), fell_back=fell_back,
        )

    async def stream(self, prompt: str, tier: str, fallback: bool = True) -> AsyncIterator[tuple[str, object]]:
        """Yield ("delta", text) events then one ("done", Result). With fallback on, the other tier is tried
        only if the chosen one fails before sending any text."""
        other = "large" if tier == "small" else "small"
        order = (tier, other) if fallback else (tier,)
        last: Exception | None = None
        for i, t in enumerate(order):
            try:
                async for event in self._run(prompt, t, fell_back=i > 0):
                    yield event
                return
            except PreTokenError as exc:
                last = exc
        raise RuntimeError(f"both tiers failed: {last}" if fallback else f"{tier} failed: {last}")

    async def generate(self, prompt: str, tier: str, fallback: bool = True) -> Result:
        result: Result | None = None
        async for kind, payload in self.stream(prompt, tier, fallback):
            if kind == "done":
                result = payload  # type: ignore[assignment]
        assert result is not None
        return result
