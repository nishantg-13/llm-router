"""Command line entry point: python -m llm_router <command>."""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time

from llm_router.config import api_key_present, load_env


def _client():
    from google import genai

    return genai.Client()  # reads GEMINI_API_KEY from the environment


def cmd_list_models(_: argparse.Namespace) -> int:
    print(f"GEMINI_API_KEY present: {'yes' if api_key_present() else 'NO (fill ~/llm-router/.env)'}")
    if not api_key_present():
        return 1
    rows = []
    client = _client()  # keep a reference: the SDK closes its HTTP client when this is collected
    for m in client.models.list():
        actions = getattr(m, "supported_actions", None) or []
        if "generateContent" in actions:
            rows.append(m.name.removeprefix("models/"))
    for name in sorted(rows):
        print(name)
    print(f"\n{len(rows)} models support generateContent for this key.")
    return 0


async def _one_call(client, model: str, prompt: str) -> tuple[float, str | None]:
    t0 = time.perf_counter()
    try:
        await client.aio.models.generate_content(model=model, contents=prompt)
        return time.perf_counter() - t0, None
    except Exception as exc:  # noqa: BLE001 - the probe reports any failure class
        return time.perf_counter() - t0, f"{type(exc).__name__}: {str(exc)[:120]}"


async def _probe(model: str, n: int) -> int:
    client = _client()
    results = await asyncio.gather(*[_one_call(client, model, "Reply with the single word: ok") for _ in range(n)])
    ok = [t for t, err in results if err is None]
    errs = [err for _, err in results if err is not None]
    print(f"model={model} concurrency={n} ok={len(ok)} failed={len(errs)}")
    if ok:
        print(f"latency s: p50={statistics.median(ok):.2f} max={max(ok):.2f}")
    for e in sorted(set(errs))[:5]:
        print(f"  error: {e}")
    return 0 if not errs else 2


def cmd_probe(args: argparse.Namespace) -> int:
    if not api_key_present():
        print("GEMINI_API_KEY missing; fill ~/llm-router/.env")
        return 1
    return asyncio.run(_probe(args.model, args.n))


def cmd_ask(args: argparse.Namespace) -> int:
    from llm_router.config import load_config
    from llm_router.runtime import build_pool

    pool = build_pool(load_config())
    result = asyncio.run(pool.generate(args.prompt, args.tier))
    print(result.text)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in result.meta().items()})
    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    from llm_router.bench.collect import collect
    from llm_router.bench.data import load_questions
    from llm_router.config import ROOT, load_config
    from llm_router.runtime import build_pool

    cfg = load_config()
    b = cfg["bench"]
    seed = args.seed if args.seed is not None else b["seed"]
    n_gsm, n_mmlu = (args.n_gsm8k, args.n_mmlu) if args.n_gsm8k is not None else (b["n_gsm8k"], b["n_mmlu"])
    questions = load_questions(n_gsm, n_mmlu, seed)
    out = ROOT / args.out
    print(f"{len(questions)} questions x 2 tiers -> {out}")
    stats = asyncio.run(collect(build_pool(cfg), questions, out, b["max_concurrency"]))
    print(stats)
    return 0 if stats["failed"] == 0 else 2


def cmd_rescore(args: argparse.Namespace) -> int:
    from llm_router.bench.collect import rescore
    from llm_router.bench.data import load_questions
    from llm_router.config import ROOT, load_config

    b = load_config()["bench"]
    seed = args.seed if args.seed is not None else b["seed"]
    n_gsm, n_mmlu = (args.n_gsm8k, args.n_mmlu) if args.n_gsm8k is not None else (b["n_gsm8k"], b["n_mmlu"])
    print(rescore(ROOT / args.out, load_questions(n_gsm, n_mmlu, seed)))
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    import json

    from llm_router.bench.data import load_questions
    from llm_router.bench.evaluate import evaluate
    from llm_router.config import ROOT, load_config
    from llm_router.runtime import build_embedder

    cfg = load_config()
    b = cfg["bench"]
    questions = load_questions(b["n_gsm8k"], b["n_mmlu"], b["seed"])
    report, summary = evaluate(questions, ROOT / args.out, build_embedder(), b["train_fraction"], b["seed"], cfg)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "report.md").write_text(report)
    (ROOT / "results" / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(report)
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from llm_router.config import load_config
    from llm_router.gateway import create_app
    from llm_router.policies import default_policies
    from llm_router.runtime import build_pool

    cfg = load_config()
    app = create_app(
        build_pool(cfg), default_policies(), default_policy=args.policy, cache_size=cfg["gateway"]["cache_size"]
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    load_env()
    p = argparse.ArgumentParser(prog="llm_router")
    sub = p.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask", help="send one prompt to a tier and print the answer with metrics")
    ask.add_argument("prompt")
    ask.add_argument("--tier", choices=["small", "large"], default="small")
    ask.set_defaults(fn=cmd_ask)
    co = sub.add_parser("collect", help="run all benchmark questions through both tiers")
    co.add_argument("--out", default="results/collect.jsonl")
    co.add_argument("--seed", type=int, default=None)
    co.add_argument("--n-gsm8k", type=int, default=None)
    co.add_argument("--n-mmlu", type=int, default=None)
    co.set_defaults(fn=cmd_collect)
    rs = sub.add_parser("rescore", help="re-grade stored responses with the current scorer (no API calls)")
    rs.add_argument("--out", default="results/collect.jsonl")
    rs.add_argument("--seed", type=int, default=None)
    rs.add_argument("--n-gsm8k", type=int, default=None)
    rs.add_argument("--n-mmlu", type=int, default=None)
    rs.set_defaults(fn=cmd_rescore)
    ev = sub.add_parser("evaluate", help="replay policies on recorded outcomes and write results/report.md")
    ev.add_argument("--out", default="results/collect.jsonl")
    ev.set_defaults(fn=cmd_evaluate)
    sv = sub.add_parser("serve", help="run the gateway")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8000)
    sv.add_argument("--policy", default="always_small")
    sv.set_defaults(fn=cmd_serve)
    sub.add_parser("list-models", help="list models this key can call").set_defaults(fn=cmd_list_models)
    pr = sub.add_parser("probe", help="fire N concurrent tiny requests to check latency and rate limits")
    pr.add_argument("--model", required=True)
    pr.add_argument("--n", type=int, default=8)
    pr.set_defaults(fn=cmd_probe)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
