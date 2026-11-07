"""FastAPI gateway: picks a tier with a policy, streams the answer over SSE, caches, and records metrics."""

from __future__ import annotations

import json
import statistics
from collections import OrderedDict

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from llm_router.policies import Policy
from llm_router.pool import Pool, Result


class Req(BaseModel):
    prompt: str
    policy: str | None = None


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    return s[min(len(s) - 1, int(q * len(s)))]


def create_app(pool: Pool, policies: dict[str, Policy], default_policy: str, cache_size: int = 256) -> FastAPI:
    app = FastAPI(title="llm-router")
    cache: OrderedDict[tuple[str, str], Result] = OrderedDict()
    log: list[Result] = []

    def pick(req: Req) -> tuple[str, str]:
        name = req.policy or default_policy
        if name not in policies:
            raise HTTPException(status_code=400, detail=f"unknown policy {name!r}; have {sorted(policies)}")
        return name, policies[name].choose(req.prompt)

    def cache_get(tier: str, prompt: str) -> Result | None:
        hit = cache.get((tier, prompt))
        if hit is None:
            return None
        cache.move_to_end((tier, prompt))
        return Result(**{**hit.__dict__, "cached": True, "cost_usd": 0.0, "ttft_s": 0.0, "total_s": 0.0})

    def remember(prompt: str, result: Result) -> None:
        if result.fell_back:
            return  # do not cache answers produced by the fallback tier under the chosen tier
        cache[(result.tier, prompt)] = result
        cache.move_to_end((result.tier, prompt))
        while len(cache) > cache_size:
            cache.popitem(last=False)

    @app.post("/v1/complete")
    async def complete(req: Req):
        policy_name, tier = pick(req)
        hit = cache_get(tier, req.prompt)
        if hit is not None:
            log.append(hit)
            return {"policy": policy_name, "text": hit.text, **hit.meta()}
        try:
            result = await pool.generate(req.prompt, tier)
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        remember(req.prompt, result)
        log.append(result)
        return {"policy": policy_name, "text": result.text, **result.meta()}

    @app.post("/v1/stream")
    async def stream(req: Req):
        policy_name, tier = pick(req)

        async def events():
            hit = cache_get(tier, req.prompt)
            if hit is not None:
                log.append(hit)
                yield f"data: {json.dumps({'delta': hit.text})}\n\n"
                yield f"event: done\ndata: {json.dumps({'policy': policy_name, **hit.meta()})}\n\n"
                return
            try:
                async for kind, payload in pool.stream(req.prompt, tier):
                    if kind == "delta":
                        yield f"data: {json.dumps({'delta': payload})}\n\n"
                    else:
                        result: Result = payload  # type: ignore[assignment]
                        remember(req.prompt, result)
                        log.append(result)
                        yield f"event: done\ndata: {json.dumps({'policy': policy_name, **result.meta()})}\n\n"
            except Exception as exc:  # noqa: BLE001 - surface any failure to the client as an SSE error event
                yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    @app.get("/metrics")
    async def metrics():
        live = [r for r in log if not r.cached]
        by_tier: dict[str, int] = {}
        for r in log:
            by_tier[r.tier] = by_tier.get(r.tier, 0) + 1
        return {
            "requests": len(log),
            "cache_hits": sum(r.cached for r in log),
            "fallbacks": sum(r.fell_back for r in log),
            "by_tier": by_tier,
            "total_cost_usd": sum(r.cost_usd for r in log),
            "mean_cost_usd": statistics.fmean(r.cost_usd for r in live) if live else None,
            "latency_p50_s": _pct([r.total_s for r in live], 0.50),
            "latency_p95_s": _pct([r.total_s for r in live], 0.95),
            "ttft_p50_s": _pct([r.ttft_s for r in live], 0.50),
            "ttft_p95_s": _pct([r.ttft_s for r in live], 0.95),
        }

    @app.get("/healthz")
    async def healthz():
        return {"ok": True, "policies": sorted(policies), "default": default_policy}

    return app
