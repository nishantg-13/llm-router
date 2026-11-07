"""Cascade v0.1 (pre-registered in docs/superpowers/specs/2026-11-08-cascade-preregistration.md).

Draw K samples from the small tier at TEMPERATURE. If the extracted answers are not all identical (or any extraction
fails), escalate to the large tier. Parameters are fixed in the pre-registration and must not be tuned here.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from llm_router.bench.data import Question
from llm_router.bench.score import extract_letter, extract_number, is_correct
from llm_router.pool import Pool

K = 3
TEMPERATURE = 0.7
RETRIES = 3


def extract(kind: str, response: str) -> str | None:
    return extract_number(response) if kind == "num" else extract_letter(response)


def should_escalate(kind: str, responses: list[str]) -> bool:
    """True unless every sample yields the same non-empty extracted answer."""
    answers = [extract(kind, r) for r in responses]
    return any(a is None for a in answers) or len(set(answers)) > 1


def load_done(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    done = set()
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("error") is None:
                done.add((rec["id"], rec["sample"]))
    return done


async def _sample(pool: Pool, q: Question, idx: int) -> dict:
    last_err = None
    for attempt in range(RETRIES):
        try:
            r = await pool.generate(q.prompt, "small", fallback=False)
            return {"id": q.id, "sample": idx, "response": r.text, "correct": is_correct(q.kind, r.text, q.gold),
                    "total_s": r.total_s, "cost_usd": r.cost_usd, "error": None}
        except Exception as exc:  # noqa: BLE001 - any failure is retried, then recorded
            last_err = f"{type(exc).__name__}: {str(exc)[:200]}"
            await asyncio.sleep(2 ** attempt)
    return {"id": q.id, "sample": idx, "error": last_err}


async def collect_samples(pool: Pool, questions: list[Question], out: Path, concurrency: int, k: int = K) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    done = load_done(out)
    todo = [(q, i) for q in questions for i in range(k) if (q.id, i) not in done]
    sem, lock = asyncio.Semaphore(concurrency), asyncio.Lock()
    stats = {"todo": len(todo), "ok": 0, "failed": 0}

    async def work(q: Question, i: int, fh) -> None:
        async with sem:
            rec = await _sample(pool, q, i)
        async with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            stats["failed" if rec["error"] else "ok"] += 1
            n = stats["ok"] + stats["failed"]
            if n % 50 == 0 or n == stats["todo"]:
                print(f"  {n}/{stats['todo']} done ({stats['failed']} failed)", flush=True)

    with out.open("a") as fh:
        await asyncio.gather(*[work(q, i, fh) for q, i in todo])
    return stats


def load_samples(path: Path) -> dict[str, dict[int, dict]]:
    table: dict[str, dict[int, dict]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("error") is None:
                table.setdefault(rec["id"], {})[rec["sample"]] = rec
    return table


@dataclass
class CascadeOutcomes:
    escalate: np.ndarray  # bool per question
    correct: np.ndarray
    cost: np.ndarray
    total: np.ndarray
    small0_correct: np.ndarray  # sample 0 alone, the same-temperature always-small baseline
    small0_cost: np.ndarray
    small0_total: np.ndarray


def cascade_outcomes(questions: list[Question], samples: dict[str, dict[int, dict]], large: dict[str, dict],
                     k: int = K) -> CascadeOutcomes:
    """Apply the pre-registered rule. `large` maps question id to the recorded large-tier record."""
    esc, cor, cost, tot, s_cor, s_cost, s_tot = [], [], [], [], [], [], []
    for q in questions:
        s = [samples[q.id][i] for i in range(k)]
        e = should_escalate(q.kind, [x["response"] for x in s])
        big = large[q.id]
        esc.append(e)
        cor.append(bool(big["correct"]) if e else bool(s[0]["correct"]))
        cost.append(sum(x["cost_usd"] for x in s) + (big["cost_usd"] if e else 0.0))
        tot.append(max(x["total_s"] for x in s) + (big["total_s"] if e else 0.0))
        s_cor.append(bool(s[0]["correct"]))
        s_cost.append(s[0]["cost_usd"])
        s_tot.append(s[0]["total_s"])
    a = np.array
    return CascadeOutcomes(a(esc), a(cor, dtype=int), a(cost), a(tot), a(s_cor, dtype=int), a(s_cost), a(s_tot))


def evaluate_cascade(questions: list[Question], collect_path: Path, samples_path: Path, cfg: dict, seed: int = 0,
                     k: int = K) -> tuple[str, dict]:
    from llm_router.bench.evaluate import (
        Arrays,
        _fmt,
        apply_choice,
        bootstrap_outcomes,
        load_table,
        paired_diff_values,
    )

    table, samples = load_table(collect_path), load_samples(samples_path)
    usable = [q for q in questions if "large" in table.get(q.id, {}) and len(samples.get(q.id, {})) >= k]
    dropped = len(questions) - len(usable)
    large = {q.id: table[q.id]["large"] for q in usable}
    o = cascade_outcomes(usable, samples, large, k)
    n = len(usable)
    nan = np.full((2, n), np.nan)
    a = Arrays(
        correct=np.array([o.small0_correct, [large[q.id]["correct"] for q in usable]], dtype=int),
        cost=np.array([o.small0_cost, [large[q.id]["cost_usd"] for q in usable]]),
        total=np.array([o.small0_total, [large[q.id]["total_s"] for q in usable]]),
        ttft=nan,
    )
    f = float(o.escalate.mean())
    rand_name = f"random_mix@{f:.2f}"
    p = {"always_small (sample 0, T=0.7)": np.zeros(n), "always_large": np.ones(n), rand_name: np.full(n, f),
         "oracle (upper bound)": (o.small0_correct == 0).astype(float)}
    outs = {name: apply_choice(a, pr) for name, pr in p.items()}
    outs["cascade (k=3, T=0.7)"] = {"correct": o.correct, "cost": o.cost, "total": o.total, "ttft": np.full(n, np.nan)}
    fracs = {name: float(np.mean(pr)) for name, pr in p.items()} | {"cascade (k=3, T=0.7)": f}
    stats = bootstrap_outcomes(outs, fracs, reference="always_large", seed=seed)
    vs_random = paired_diff_values(o.correct - outs[rand_name]["correct"], seed=seed)

    # Post-hoc (added after seeing the primary result): a random mix that spends the same money as the cascade.
    c_small, c_large = float(a.cost[0].mean()), float(a.cost[1].mean())
    f_match = float(np.clip((float(o.cost.mean()) - c_small) / (c_large - c_small), 0.0, 1.0))
    matched = apply_choice(a, np.full(n, f_match))
    vs_cost_matched = paired_diff_values(o.correct - matched["correct"], seed=seed)

    acc_mask = ~o.escalate
    diag = {
        "n_accepted": int(acc_mask.sum()), "n_escalated": int(o.escalate.sum()),
        "accepted_small_acc": float(o.small0_correct[acc_mask].mean()) if acc_mask.any() else None,
        "escalated_small0_acc": float(o.small0_correct[o.escalate].mean()) if o.escalate.any() else None,
        "small0_failures_total": int((o.small0_correct == 0).sum()),
        "small0_failures_caught": int(((o.small0_correct == 0) & o.escalate).sum()),
    }

    m = cfg["models"]
    L = ["# Cascade v0.1 results (exploratory; rule pre-registered in `docs/superpowers/specs/2026-11-08-cascade-preregistration.md`)\n"]
    L.append(f"- Small `{m['small']['name']}` sampled k={k} times at temperature 0.7; escalate to large `{m['large']['name']}` unless all extracted answers agree")
    L.append(f"- {n} questions scored ({dropped} dropped for missing data). No component is trained, so there is no train/test split")
    L.append("- 95% paired bootstrap intervals over questions. Cost counts all small samples plus the large call when escalated; latency is the slowest sample plus the large call when escalated\n")
    L.append("| policy | % escalated to large | accuracy | Δ acc vs always-large | cost / 1k req (USD) | cost as % of always-large | latency p50 (s) | latency p95 (s) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, s in stats.items():
        p50 = _fmt(s["total_p50"], 2) if "total_p50" in s else "n/a"
        p95 = _fmt(s["total_p95"], 2) if "total_p95" in s else "n/a"
        L.append(f"| {name} | {s['frac_large'] * 100:.0f}% | {_fmt(s['accuracy'])} | {_fmt(s['acc_delta_vs_ref'])} | "
                 f"{_fmt(s['cost_per_1k_usd'], 3)} | {_fmt(s['cost_pct_of_ref'], 1)} | {p50} | {p95} |")
    lo, hi = vs_random[1:]
    verdict = "above zero: disagreement beats random escalation" if lo > 0 else (
        "below zero" if hi < 0 else "includes zero: not distinguishable from random escalation")
    L.append("\n## Primary pre-registered test\n")
    L.append(f"Cascade accuracy minus a random mix escalating the same {f * 100:.0f}% of questions: {_fmt(vs_random)}. The interval {verdict}.")
    c = stats["cascade (k=3, T=0.7)"]
    L.append(f"\nCascade cost was {_fmt(c['cost_pct_of_ref'], 1)} % of always-large, with accuracy {_fmt(c['accuracy'])} "
             f"versus always-large {_fmt(stats['always_large']['accuracy'])}.")
    m_lo, m_hi = vs_cost_matched[1:]
    mv = "includes zero: not distinguishable" if m_lo <= 0 <= m_hi else ("above zero" if m_lo > 0 else "below zero")
    L.append("\n## Post-hoc cost-matched comparison (added AFTER seeing the primary result; not pre-registered)\n")
    L.append(f"A random mix that sends {f_match * 100:.0f}% of questions to the large model spends the same average money as the cascade "
             f"(matched on point estimates; the cascade pays for three small samples on every request). Cascade accuracy minus that "
             f"cost-matched random mix: {_fmt(vs_cost_matched)}. The interval {mv}.")
    L.append("\n## Does disagreement flag the small model's failures?\n")
    acc_a = diag["accepted_small_acc"]
    acc_e = diag["escalated_small0_acc"]
    L.append(f"- Accepted (all samples agreed): {diag['n_accepted']} questions, small-model accuracy {acc_a:.3f}" if acc_a is not None else "- No questions were accepted")
    L.append(f"- Escalated (samples disagreed or no answer extracted): {diag['n_escalated']} questions, sample-0 accuracy {acc_e:.3f}" if acc_e is not None else "- No questions were escalated")
    L.append(f"- Of {diag['small0_failures_total']} sample-0 failures, {diag['small0_failures_caught']} were escalated")
    L.append("\n## Limitations\n")
    L.append("- The rule was chosen after seeing that embedding routers failed on these same questions, so this is not a clean out-of-sample test.")
    L.append("- Same 404 questions, same two models, exact-match tasks only; large-model outcomes are replayed from the v0 run.")
    L.append("- The cascade spends three small calls on every request, so it only saves money if small calls are cheap relative to the large call.")
    return "\n".join(L) + "\n", {"stats": stats, "vs_random_acc_diff": vs_random, "vs_cost_matched_acc_diff": vs_cost_matched,
                                   "cost_matched_frac_large": f_match, "diagnostics": diag, "n": n, "dropped": dropped}
