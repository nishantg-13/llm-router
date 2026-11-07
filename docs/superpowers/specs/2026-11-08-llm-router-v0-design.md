# llm-router v0: design and task list

Status: approved by the author on 2026-11-08 (overnight v0). Results come only from `results/report.md`.

## Goal
A gateway that routes each LLM request to a small or a large model and measures, with real data, the cost, latency
and quality trade-off of different routing policies.

## Components
- `llm_router/pool.py`: model pool (small, large) over the Gemini API, streaming, token counts, timeouts, fallback.
- `llm_router/gateway.py`: FastAPI app, SSE streaming, in-process LRU cache, per-request metrics.
- `llm_router/policies.py`: always-small, always-large, rules router, learned router (bge-small + logistic regression).
- `llm_router/bench/`: dataset loading (GSM8K, MMLU slice), collection run, replay evaluation, report.
- `llm_router/cli.py`: `list-models`, `probe`, `collect`, `evaluate`, `report`, `serve`.

## Evaluation
- About 200 questions: GSM8K (exact match on final number), MMLU slice (letter match). No LLM judge.
- Collect once: both models on every question; record correctness, latency, TTFT, tokens.
- Replay: each policy is scored offline against recorded outcomes. 100 questions train the learned router, 100 are held out.
- Report: accuracy vs cost per policy with 95% bootstrap CIs, p50/p95 latency and TTFT.
- Prices live in `config.yaml`, entered by the author from the provider's price page. Never typed from memory.

## Out of scope for v0
Load test, Redis, cascade and budget-aware policies, Ollama, voice, integrations.

## Tasks
0. Scaffold, config, `list-models`, `probe`.
1. Pool, streaming gateway, metrics, fallback, tests.
2. Dataset loaders and the collection run.
3. Policies, replay evaluation, report. Freeze scope here.
4. Tests, README (states v0 status honestly), CI workflow.
