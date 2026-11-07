"""Run every question through both tiers once and record the outcome. Resumable: finished (id, tier) pairs are skipped."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from llm_router.bench.data import Question
from llm_router.bench.score import is_correct
from llm_router.pool import TIERS, Pool

RETRIES = 3


def load_done(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("error") is None:
                done.add((rec["id"], rec["tier"]))
    return done


async def _one(pool: Pool, q: Question, tier: str) -> dict:
    last_err = None
    for attempt in range(RETRIES):
        try:
            r = await pool.generate(q.prompt, tier, fallback=False)
            return {
                "id": q.id, "tier": tier, "model": r.model, "response": r.text,
                "correct": is_correct(q.kind, r.text, q.gold), "ttft_s": r.ttft_s, "total_s": r.total_s,
                "input_tokens": r.input_tokens, "output_tokens": r.output_tokens, "cost_usd": r.cost_usd,
                "error": None,
            }
        except Exception as exc:  # noqa: BLE001 - any failure is retried, then recorded in the output
            last_err = f"{type(exc).__name__}: {str(exc)[:200]}"
            await asyncio.sleep(2 ** attempt)
    return {"id": q.id, "tier": tier, "error": last_err}


async def collect(pool: Pool, questions: list[Question], out: Path, concurrency: int) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    todo = [(q, t) for q in questions for t in TIERS if (q.id, t) not in done]
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    stats = {"todo": len(todo), "ok": 0, "failed": 0}

    async def work(q: Question, tier: str, fh) -> None:
        async with sem:
            rec = await _one(pool, q, tier)
        async with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            stats["failed" if rec["error"] else "ok"] += 1
            n = stats["ok"] + stats["failed"]
            if n % 20 == 0 or n == stats["todo"]:
                print(f"  {n}/{stats['todo']} done ({stats['failed']} failed)", flush=True)

    with out.open("a") as fh:
        await asyncio.gather(*[work(q, t, fh) for q, t in todo])
    return stats


def rescore(path: Path, questions: list[Question]) -> dict:
    """Recompute `correct` for every stored response from the current scorer. No API calls."""
    by_id = {q.id: q for q in questions}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    changed = 0
    for rec in rows:
        if rec.get("error") is None and rec["id"] in by_id:
            q = by_id[rec["id"]]
            new = is_correct(q.kind, rec["response"], q.gold)
            changed += new != rec["correct"]
            rec["correct"] = new
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {"rows": len(rows), "changed": changed}
