"""Replay evaluation: score routing policies offline against recorded per-tier outcomes, with paired bootstrap CIs."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from llm_router.bench.data import Question
from llm_router.policies import LearnedRouter, RulesRouter

THRESHOLDS = (0.5, 0.7, 0.8, 0.9, 0.95)
HEADLINE_THRESHOLD = 0.9  # fixed before any held-out result was computed
BOOTSTRAPS = 2000
TIER_ROW = {"small": 0, "large": 1}


@dataclass
class Arrays:
    """Recorded outcomes, shape (2, n): row 0 = small tier, row 1 = large tier."""

    correct: np.ndarray
    cost: np.ndarray
    total: np.ndarray
    ttft: np.ndarray


def load_table(path: Path) -> dict[str, dict[str, dict]]:
    table: dict[str, dict[str, dict]] = {}
    for line in path.read_text().splitlines():
        if line.strip():
            rec = json.loads(line)
            if rec.get("error") is None:
                table.setdefault(rec["id"], {})[rec["tier"]] = rec
    return table


def split_ids(questions: list[Question], train_fraction: float, seed: int) -> tuple[list[Question], list[Question]]:
    """Stratified by source, seeded, so train and test have the same mix of GSM8K and MMLU."""
    train: list[Question] = []
    test: list[Question] = []
    for source in sorted({q.source for q in questions}):
        group = [q for q in questions if q.source == source]
        random.Random(f"{seed}-{source}").shuffle(group)
        cut = int(train_fraction * len(group))
        train += group[:cut]
        test += group[cut:]
    return train, test


def to_arrays(questions: list[Question], table: dict[str, dict[str, dict]]) -> Arrays:
    def col(field: str, cast=float) -> np.ndarray:
        return np.array([[cast(table[q.id][t][field]) for q in questions] for t in ("small", "large")])

    return Arrays(correct=col("correct", int), cost=col("cost_usd"), total=col("total_s"), ttft=col("ttft_s"))


def apply_choice(a: Arrays, p_large: np.ndarray) -> dict[str, np.ndarray]:
    """Outcome of a policy. p_large is the probability of routing each question to the large tier. Hard policies
    use 0 or 1. Fractional values give expected correctness and cost; latency is undefined (NaN) for those."""
    p = np.asarray(p_large, dtype=float)
    mix = lambda m: (1 - p) * m[0] + p * m[1]
    hard = np.isin(p, (0.0, 1.0))
    lat = lambda m: np.where(hard, mix(m), np.nan)
    return {"correct": mix(a.correct), "cost": mix(a.cost), "total": lat(a.total), "ttft": lat(a.ttft)}


def _ci(x: np.ndarray) -> tuple[float, float]:
    lo, hi = np.nanpercentile(x, [2.5, 97.5])
    return float(lo), float(hi)


def bootstrap(a: Arrays, policies: dict[str, np.ndarray], reference: str, B: int = BOOTSTRAPS, seed: int = 0) -> dict:
    """Paired bootstrap over questions for routing policies expressed as P(large) per question."""
    outs = {name: apply_choice(a, p) for name, p in policies.items()}
    fracs = {name: float(np.mean(p)) for name, p in policies.items()}
    return bootstrap_outcomes(outs, fracs, reference, B, seed)


def bootstrap_outcomes(outs: dict[str, dict], fracs: dict[str, float], reference: str, B: int = BOOTSTRAPS, seed: int = 0) -> dict:
    """Paired bootstrap over questions. Every policy is scored on the same resampled questions, so differences
    against `reference` are paired. `outs[name]` holds per-question arrays: correct, cost, and optionally total and
    ttft. Latency fields are reported only when they are fully defined. Returns point estimates and 95% intervals."""
    n = len(outs[reference]["correct"])
    idx = np.random.default_rng(seed).integers(0, n, size=(B, n))
    ref = outs[reference]
    ref_acc_b, ref_cost_b = ref["correct"][idx].mean(1), ref["cost"][idx].mean(1)
    result: dict[str, dict] = {}
    for name, o in outs.items():
        acc_b, cost_b = o["correct"][idx].mean(1), o["cost"][idx].mean(1)
        row = {
            "n": n,
            "frac_large": fracs[name],
            "accuracy": (float(o["correct"].mean()), *_ci(acc_b)),
            "cost_per_1k_usd": (float(o["cost"].mean() * 1000), *_ci(cost_b * 1000)),
            "acc_delta_vs_ref": (float(o["correct"].mean() - ref["correct"].mean()), *_ci(acc_b - ref_acc_b)),
            "cost_pct_of_ref": (float(o["cost"].mean() / ref["cost"].mean() * 100), *_ci(cost_b / ref_cost_b * 100)),
        }
        for key in ("total", "ttft"):
            if key in o and not np.isnan(o[key]).any():
                row[f"{key}_p50"] = (float(np.percentile(o[key], 50)), *_ci(np.percentile(o[key][idx], 50, axis=1)))
        if "total" in o and not np.isnan(o["total"]).any():
            row["total_p95"] = (float(np.percentile(o["total"], 95)), *_ci(np.percentile(o["total"][idx], 95, axis=1)))
        result[name] = row
    return result


def paired_diff_values(d: np.ndarray, B: int = BOOTSTRAPS, seed: int = 0) -> tuple[float, float, float]:
    """Mean of per-question differences d with a bootstrap interval over questions."""
    idx = np.random.default_rng(seed).integers(0, len(d), size=(B, len(d)))
    return float(d.mean()), *_ci(d[idx].mean(1))


def paired_diff(a: Arrays, p1: np.ndarray, p2: np.ndarray, B: int = BOOTSTRAPS, seed: int = 0) -> tuple[float, float, float]:
    """Accuracy of policy 1 minus policy 2 with a paired bootstrap interval."""
    return paired_diff_values(apply_choice(a, p1)["correct"] - apply_choice(a, p2)["correct"], B, seed)


def _fmt(t: tuple[float, float, float], digits: int = 3, scale: float = 1.0, suffix: str = "") -> str:
    v, lo, hi = (x * scale for x in t)
    return f"{v:.{digits}f}{suffix} [{lo:.{digits}f}, {hi:.{digits}f}]"


def source_headroom(questions: list[Question], table: dict[str, dict[str, dict]]) -> list[tuple[str, int, float, float]]:
    rows = []
    for source in sorted({q.source for q in questions}):
        qs = [q for q in questions if q.source == source]
        rows.append((source, len(qs), *[float(np.mean([table[q.id][t]["correct"] for q in qs])) for t in ("small", "large")]))
    return rows


def measure_overhead_ms(policy, prompts: list[str], reps: int = 30) -> float:
    sample = (prompts * (reps // max(1, len(prompts)) + 1))[:reps]
    policy.choose(sample[0])  # warm-up
    t0 = time.perf_counter()
    for p in sample:
        policy.choose(p)
    return (time.perf_counter() - t0) / len(sample) * 1000


def auc_with_ci(p_ok: np.ndarray, small_correct: np.ndarray, B: int = 1000, seed: int = 0) -> tuple[float, float, float] | None:
    """ROC-AUC of the router's 'small model will be right' score. 0.5 means no skill. None if only one class."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(small_correct, dtype=int)
    if y.min() == y.max():
        return None
    point = float(roc_auc_score(y, p_ok))
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            aucs.append(roc_auc_score(y[i], p_ok[i]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return point, float(lo), float(hi)


def ranked_frontier(a: Arrays, p_ok: np.ndarray, fractions: tuple[float, ...], seed: int = 0) -> list[dict]:
    """Threshold-free: send the k least-confident questions (lowest p_ok) to the large tier and compare against a
    random mix sending the same fraction. Paired bootstrap on the difference."""
    n = a.correct.shape[1]
    order = np.argsort(p_ok, kind="stable")
    rows = []
    for f in fractions:
        k = round(f * n)
        learned = np.zeros(n)
        learned[order[:k]] = 1.0
        rand = np.full(n, k / n)
        s = bootstrap(a, {"always_large": np.ones(n), "learned": learned, "random": rand}, "always_large", B=1000, seed=seed)
        rows.append({
            "frac": k / n, "learned_acc": s["learned"]["accuracy"], "random_acc": s["random"]["accuracy"],
            "diff_vs_random": paired_diff(a, learned, rand, B=1000, seed=seed),
            "cost_pct_of_large": s["learned"]["cost_pct_of_ref"],
        })
    return rows


def evaluate(questions: list[Question], path: Path, embed, train_fraction: float, seed: int, cfg: dict) -> tuple[str, dict]:
    table = load_table(path)
    usable = [q for q in questions if {"small", "large"} <= set(table.get(q.id, {}))]
    dropped = len(questions) - len(usable)
    train, test = split_ids(usable, train_fraction, seed)
    a_test = to_arrays(test, table)

    learned = LearnedRouter(embed).fit([q.prompt for q in train], [bool(table[q.id]["small"]["correct"]) for q in train])
    p_ok = learned.p_small_ok([q.prompt for q in test])
    rules = RulesRouter()

    n = len(test)
    policies: dict[str, np.ndarray] = {
        "always_small": np.zeros(n),
        "always_large": np.ones(n),
        "rules": np.array([rules.choose(q.prompt) == "large" for q in test], dtype=float),
    }
    for thr in THRESHOLDS:
        policies[f"learned@{thr}"] = (p_ok < thr).astype(float)
    headline = f"learned@{HEADLINE_THRESHOLD}"
    frac = float(policies[headline].mean())
    policies[f"random_mix@{frac:.2f}"] = np.full(n, frac)
    policies["oracle (upper bound)"] = (a_test.correct[0] == 0).astype(float)

    stats = bootstrap(a_test, policies, reference="always_large", seed=seed)
    vs_random = paired_diff(a_test, policies[headline], policies[f"random_mix@{frac:.2f}"], seed=seed)
    headroom = source_headroom(usable, table)
    overhead = {
        "rules_ms": measure_overhead_ms(rules, [q.prompt for q in test]),
        "learned_ms": measure_overhead_ms(LearnedRouter(embed, HEADLINE_THRESHOLD).fit(
            [q.prompt for q in train], [bool(table[q.id]["small"]["correct"]) for q in train]), [q.prompt for q in test]),
    }

    y_test = a_test.correct[0]
    sources = np.array([q.source for q in test])
    explore = {
        "n_small_wrong_test": int((y_test == 0).sum()),
        "n_small_wrong_train": int(sum(not table[q.id]["small"]["correct"] for q in train)),
        "auc_overall": auc_with_ci(p_ok, y_test, seed=seed),
        "auc_by_source": {s: auc_with_ci(p_ok[sources == s], y_test[sources == s], seed=seed) for s in sorted(set(sources))},
        "frontier": ranked_frontier(a_test, p_ok, (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0), seed=seed),
    }

    hidden = {}
    for thr in THRESHOLDS:
        name = f"learned@{thr}"
        if name == headline:
            continue
        for ref in ("always_small", "always_large"):
            if np.array_equal(policies[name], policies[ref]):
                hidden[name] = ref
                break
    p_range = (float(p_ok.min()), float(p_ok.max()))

    report = render(cfg, len(usable), dropped, len(train), len(test), stats, headroom, overhead, headline, frac, vs_random,
                    explore, hidden=hidden, p_range=p_range)
    return report, {"stats": stats, "headroom": headroom, "overhead_ms": overhead, "n_train": len(train), "n_test": len(test),
                    "dropped": dropped, "vs_random_acc_diff": vs_random, "explore": explore}


def render_summary(stats: dict, headline: str, vs_random: tuple, explore: dict, frac: float) -> list[str]:
    """Plain-language summary computed from the numbers below; nothing here is typed by hand."""
    sm, lg, orc, h = stats["always_small"], stats["always_large"], stats["oracle (upper bound)"], stats[headline]
    auc = explore["auc_overall"]
    r_lo, r_hi = vs_random[1:]
    rv = "indistinguishable from a random mix" if r_lo <= 0 <= r_hi else ("worse than a random mix" if r_hi < 0 else "better than a random mix")
    a_lo, a_hi = (auc[1], auc[2]) if auc else (None, None)
    av = "no detectable skill (interval includes 0.5)" if auc and a_lo <= 0.5 <= a_hi else ("some skill" if auc and a_lo > 0.5 else "n/a")
    return [
        "## Summary (computed from the numbers below)\n",
        (f"- **Headroom exists.** Small model accuracy {sm['accuracy'][0]:.3f}, large {lg['accuracy'][0]:.3f}; the small model costs "
         f"{sm['cost_pct_of_ref'][0]:.0f}% of the large model per request."),
        (f"- **Routing could pay off in principle.** An oracle that escalates only when the small model fails reaches accuracy "
         f"{orc['accuracy'][0]:.3f} at {orc['cost_pct_of_ref'][0]:.0f}% of the large model's cost, escalating {orc['frac_large'] * 100:.0f}% of questions."),
        (f"- **The routers did not capture it.** The pre-fixed learned router ({headline}) escalated {h['frac_large'] * 100:.0f}% and was "
         f"{rv}; its ROC-AUC shows {av}."),
        "",
    ]


def render_explore(x: dict) -> list[str]:
    L = ["\n## Exploratory diagnostics (added AFTER seeing the headline result; not pre-specified)\n"]
    L.append(f"Small-model failures: {x['n_small_wrong_train']} in the training half, {x['n_small_wrong_test']} in the test half. "
             "Few failures means little signal to learn from and wide intervals.\n")
    L.append("ROC-AUC of the learned router's score for 'small model is correct' (0.5 = no skill):\n")
    L.append("| subset | ROC-AUC |\n|---|---|")
    overall = x["auc_overall"]
    L.append(f"| all test questions | {_fmt(overall, 3) if overall else 'n/a'} |")
    for source, v in x["auc_by_source"].items():
        L.append(f"| {source} only | {_fmt(v, 3) if v else 'n/a (one class)'} |")
    L.append("\nThreshold-free frontier: send the k% least-confident questions to the large model, compared with a random mix at the same k:\n")
    L.append("| % to large | learned acc | random-mix acc | learned minus random | cost as % of always-large |")
    L.append("|---|---|---|---|---|")
    for r in x["frontier"]:
        L.append(f"| {r['frac'] * 100:.0f}% | {_fmt(r['learned_acc'])} | {_fmt(r['random_acc'])} | "
                 f"{_fmt(r['diff_vs_random'])} | {_fmt(r['cost_pct_of_large'], 1)} |")
    return L


def render(cfg, n_usable, dropped, n_train, n_test, stats, headroom, overhead, headline, frac, vs_random, explore,
           hidden=(), p_range=None) -> str:
    m = cfg["models"]
    L = []
    L.append("# Results (generated by `python -m llm_router evaluate`; never edited by hand)\n")
    L.append(f"- Small tier: `{m['small']['name']}` (${m['small']['usd_per_1m_input']}/1M in, ${m['small']['usd_per_1m_output']}/1M out)")
    L.append(f"- Large tier: `{m['large']['name']}` (${m['large']['usd_per_1m_input']}/1M in, ${m['large']['usd_per_1m_output']}/1M out)")
    L.append(f"- Questions: {n_usable} usable ({dropped} dropped for missing outcomes); train {n_train} / held-out test {n_test}, stratified by source")
    L.append("- Quality is exact match (number or letter). No LLM judge. Intervals are 95% paired bootstrap over test questions\n")

    L += render_summary(stats, headline, vs_random, explore, frac)

    L.append("## Headroom: how much better is the large model?\n")
    L.append("| source | n | small acc | large acc |\n|---|---|---|---|")
    for source, k, s, lg in headroom:
        L.append(f"| {source} | {k} | {s:.3f} | {lg:.3f} |")

    L.append("\n## Policies on the held-out test set\n")
    L.append("| policy | % to large | accuracy | Δ acc vs always-large | cost / 1k req (USD) | cost as % of always-large | latency p50 (s) | latency p95 (s) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, s in stats.items():
        if name in hidden:
            continue
        p50 = _fmt(s["total_p50"], 2) if "total_p50" in s else "n/a"
        p95 = _fmt(s["total_p95"], 2) if "total_p95" in s else "n/a"
        L.append(f"| {name} | {s['frac_large'] * 100:.0f}% | {_fmt(s['accuracy'])} | {_fmt(s['acc_delta_vs_ref'])} | "
                 f"{_fmt(s['cost_per_1k_usd'], 3)} | {_fmt(s['cost_pct_of_ref'], 1)} | {p50} | {p95} |")

    if hidden:
        groups = {}
        for name, same_as in hidden.items():
            groups.setdefault(same_as, []).append(name)
        parts = [f"{', '.join(v)} (identical to {k})" for k, v in groups.items()]
        L.append("\nRows collapsed because their routing decisions were identical to another row: " + "; ".join(parts) + ". "
                 f"The learned router's scores on the test set ranged only from {p_range[0]:.2f} to {p_range[1]:.2f}, so the threshold sweep "
                 "carries little information. The full table is in `results/summary.json`.")

    h = stats[headline]
    L.append(f"\n## Headline operating point: {headline} (fixed before looking at held-out results)\n")
    d_lo, d_hi = h["acc_delta_vs_ref"][1:]
    verdict = ("is not distinguishable from zero" if d_lo <= 0 <= d_hi
               else "is below zero" if d_hi < 0 else "is above zero")
    L.append(f"- Routed {h['frac_large'] * 100:.0f}% of test questions to the large model.")
    L.append(f"- Accuracy {_fmt(h['accuracy'])} versus always-large {_fmt(stats['always_large']['accuracy'])}; "
             f"the paired difference {_fmt(h['acc_delta_vs_ref'])} {verdict} at 95%.")
    L.append(f"- Cost was {_fmt(h['cost_pct_of_ref'], 1)} % of always-large.")
    r_lo, r_hi = vs_random[1:]
    rv = ("not distinguishable from zero" if r_lo <= 0 <= r_hi else "below zero" if r_hi < 0 else "above zero")
    L.append(f"- Versus a random mix sending the same {frac * 100:.0f}% to the large model, accuracy differs by "
             f"{_fmt(vs_random)}, which is {rv}. This is the test of whether the router has real skill.")

    L += render_explore(explore)

    L.append("\n## Router overhead (this machine, per request, single prompt)\n")
    L.append(f"- rules: {overhead['rules_ms']:.3f} ms; learned (embedding + logistic regression): {overhead['learned_ms']:.1f} ms")

    L.append("\n## Limitations\n")
    L.append("- Small benchmark (hundreds of questions), two model tiers from one provider, and a replay evaluation: each question "
             "was run once per tier, and policies are scored on those recorded outcomes.")
    L.append("- Latencies come from the collection run under 8-way concurrency, so they include some contention. They are not a load test.")
    L.append("- Only exact-match tasks (math and multiple choice). Results may not transfer to open-ended or conversational traffic.")
    L.append("- Several thresholds are shown for context; only the headline threshold was fixed in advance.")
    L.append("- Prices are list prices copied into `config.yaml` on the date noted there. Costs scale with them.")
    return "\n".join(L) + "\n"
