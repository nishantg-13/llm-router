# Cascade v0.1: pre-registration (written before any cascade sample was collected)

Date: 2026-11-08. Status: exploratory follow-up to v0, motivated by the v0 result that prompt-embedding routers
showed no skill (ROC-AUC 0.509, interval including 0.5).

## Question
Can the small model's own disagreement across repeated samples tell us when to escalate to the large model, saving
cost without a large accuracy loss?

## Rule (fixed now; no parameter will be changed after seeing results)
- Draw **k = 3** independent samples from the small tier at **temperature 0.7**, same prompts as v0.
- Extract the final answer from each sample with the existing scorer (number or letter).
- **Escalate to the large tier if the three extracted answers are not all identical, or if any extraction fails.**
  Otherwise accept the (unanimous) small answer.
- No component is trained, so there is no train/test split; all 404 questions are scored.
- Escalated questions use the large model's already-recorded outcome from the v0 collection run.

## Accounting (fixed now)
- Cost = cost of all 3 small samples, plus the large model's recorded cost when escalated.
- Latency = the slowest of the 3 small samples (they run in parallel), plus the large model's recorded latency when escalated.

## Comparisons and statistics
- Baselines on the same questions: always-small (sample 0 of the cascade run, so same temperature), always-large,
  a random mix that escalates the same fraction of questions, and an oracle (escalate exactly when sample 0 is wrong).
- 95% paired bootstrap intervals over questions, 2000 resamples, same code as v0.
- **Primary test of skill:** cascade accuracy minus the random-mix accuracy at the same escalation fraction.
  An interval above zero means the disagreement signal beats random escalation; an interval containing zero means it does not.
- Secondary: cascade cost as a percentage of always-large, and accuracy difference versus always-large.

## Commitments
- The result will be published whatever it shows, in `results/cascade_report.md`, labeled exploratory.
- No second cascade variant (different k, temperature, or rule) will be run and reported as the headline.
  If another variant is tried, it is reported as a separate exploratory run next to this one.

## Known limitations, stated in advance
- The rule was chosen after seeing that embedding-based routers failed on these same questions, so it is not a clean
  out-of-sample test. It was chosen from the general self-consistency idea, not by tuning on these questions.
- Same 404 questions, same two models, exact-match tasks only, replay of recorded large-model outcomes.
- Cascade costs 3 small calls on every request, so it only saves money if small calls are cheap relative to the large call.
