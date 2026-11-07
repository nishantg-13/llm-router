# Cascade v0.1 results (exploratory; rule pre-registered in `docs/superpowers/specs/2026-11-08-cascade-preregistration.md`)

- Small `gemini-2.5-flash-lite` sampled k=3 times at temperature 0.7; escalate to large `gemini-3.8-flash` unless all extracted answers agree
- 404 questions scored (0 dropped for missing data). No component is trained, so there is no train/test split
- 95% paired bootstrap intervals over questions. Cost counts all small samples plus the large call when escalated; latency is the slowest sample plus the large call when escalated

| policy | % escalated to large | accuracy | Δ acc vs always-large | cost / 1k req (USD) | cost as % of always-large | latency p50 (s) | latency p95 (s) |
|---|---|---|---|---|---|---|---|
| always_small (sample 0, T=0.7) | 0% | 0.894 [0.864, 0.923] | -0.087 [-0.116, -0.057] | 0.249 [0.210, 0.294] | 16.2 [13.5, 19.2] | 1.55 [1.51, 1.63] | 5.91 [4.08, 7.98] |
| always_large | 100% | 0.980 [0.965, 0.993] | 0.000 [0.000, 0.000] | 1.534 [1.420, 1.669] | 100.0 [100.0, 100.0] | 4.02 [3.85, 4.19] | 8.25 [7.64, 10.95] |
| random_mix@0.11 | 11% | 0.903 [0.876, 0.929] | -0.077 [-0.104, -0.051] | 0.385 [0.345, 0.433] | 25.1 [22.7, 27.8] | n/a | n/a |
| oracle (upper bound) | 11% | 0.988 [0.975, 0.998] | 0.007 [0.000, 0.017] | 0.415 [0.337, 0.505] | 27.0 [22.4, 32.0] | 1.63 [1.55, 1.73] | 7.13 [5.52, 8.41] |
| cascade (k=3, T=0.7) | 11% | 0.950 [0.931, 0.970] | -0.030 [-0.050, -0.012] | 1.061 [0.880, 1.263] | 69.1 [59.3, 79.3] | 1.76 [1.66, 1.90] | 13.88 [9.85, 18.13] |

## Primary pre-registered test

Cascade accuracy minus a random mix escalating the same 11% of questions: 0.048 [0.027, 0.069]. The interval above zero: disagreement beats random escalation.

Cascade cost was 69.1 [59.3, 79.3] % of always-large, with accuracy 0.950 [0.931, 0.970] versus always-large 0.980 [0.965, 0.993].

## Post-hoc cost-matched comparison (added AFTER seeing the primary result; not pre-registered)

A random mix that sends 63% of questions to the large model spends the same average money as the cascade (matched on point estimates; the cascade pays for three small samples on every request). Cascade accuracy minus that cost-matched random mix: 0.002 [-0.013, 0.017]. The interval includes zero: not distinguishable.

## Does disagreement flag the small model's failures?

- Accepted (all samples agreed): 361 questions, small-model accuracy 0.950
- Escalated (samples disagreed or no answer extracted): 43 questions, sample-0 accuracy 0.419
- Of 43 sample-0 failures, 25 were escalated

## Limitations

- The rule was chosen after seeing that embedding routers failed on these same questions, so this is not a clean out-of-sample test.
- Same 404 questions, same two models, exact-match tasks only; large-model outcomes are replayed from the v0 run.
- The cascade spends three small calls on every request, so it only saves money if small calls are cheap relative to the large call.
