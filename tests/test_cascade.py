import numpy as np
import pytest

from llm_router.bench.cascade import cascade_outcomes, should_escalate
from llm_router.bench.data import Question


def test_unanimous_numeric_answers_do_not_escalate():
    assert not should_escalate("num", ["so 7.\nAnswer: 7", "Answer: 7.0", "final: \\boxed{7}"])


def test_any_disagreement_escalates():
    assert should_escalate("num", ["Answer: 7", "Answer: 7", "Answer: 8"])
    assert should_escalate("mcq", ["Answer: A", "Answer: B", "Answer: A"])


def test_failed_extraction_escalates():
    assert should_escalate("mcq", ["Answer: A", "I am not sure", "Answer: A"])
    assert should_escalate("num", ["no digits", "no digits", "no digits"])


def _q(i, kind="num"):
    return Question(f"q{i}", "gsm8k", "x", kind, "p", "7")


def _sample(resp, correct, cost=1.0, total=2.0):
    return {"response": resp, "correct": correct, "cost_usd": cost, "total_s": total}


def test_cascade_accounting_matches_the_preregistration():
    qs = [_q(0), _q(1)]
    samples = {
        "q0": {0: _sample("Answer: 7", True, 1, 2), 1: _sample("Answer: 7", True, 1, 3), 2: _sample("Answer: 7", True, 1, 2.5)},
        "q1": {0: _sample("Answer: 7", True, 1, 2), 1: _sample("Answer: 9", False, 1, 4), 2: _sample("Answer: 7", True, 1, 1)},
    }
    large = {"q0": {"correct": 1, "cost_usd": 10.0, "total_s": 8.0}, "q1": {"correct": 1, "cost_usd": 10.0, "total_s": 6.0}}
    o = cascade_outcomes(qs, samples, large)
    assert o.escalate.tolist() == [False, True]
    assert o.correct.tolist() == [1, 1]  # q0 accepted from the small samples, q1 taken from the large tier
    assert o.cost.tolist() == pytest.approx([3.0, 13.0])  # three small calls, plus the large call when escalated
    assert o.total.tolist() == pytest.approx([3.0, 10.0])  # slowest sample (parallel), plus large latency when escalated
    assert o.small0_correct.tolist() == [1, 1]


def test_escalated_wrong_small_answer_is_replaced_by_large_outcome():
    qs = [_q(0)]
    samples = {"q0": {0: _sample("Answer: 3", False), 1: _sample("Answer: 4", False), 2: _sample("Answer: 3", False)}}
    o = cascade_outcomes(qs, samples, {"q0": {"correct": 1, "cost_usd": 5.0, "total_s": 1.0}})
    assert o.correct.tolist() == [1] and o.escalate.tolist() == [True]
    assert isinstance(o.cost, np.ndarray)


def test_evaluate_cascade_end_to_end_on_synthetic_files(tmp_path):
    import json

    from llm_router.bench.cascade import evaluate_cascade

    qs = [Question(f"q{i}", "gsm8k", "x", "num", "p", "7") for i in range(8)]
    collect = tmp_path / "collect.jsonl"
    samples = tmp_path / "samples.jsonl"
    with collect.open("w") as c, samples.open("w") as s:
        for i, q in enumerate(qs):
            small_ok = i % 4 != 0  # q0 and q4: the small model is wrong
            for tier, ok, cost, tot in (("small", small_ok, 1.0, 2.0), ("large", True, 10.0, 6.0)):
                c.write(json.dumps({"id": q.id, "tier": tier, "correct": ok, "cost_usd": cost, "total_s": tot,
                                    "ttft_s": 1.0, "error": None}) + "\n")
            for k in range(3):
                answer = "7" if small_ok else str(3 + k)  # wrong questions disagree across samples, so they escalate
                s.write(json.dumps({"id": q.id, "sample": k, "response": f"Answer: {answer}", "correct": small_ok,
                                    "cost_usd": 1.0, "total_s": 2.0, "error": None}) + "\n")
    cfg = {"models": {"small": {"name": "s", "usd_per_1m_input": 1, "usd_per_1m_output": 1},
                      "large": {"name": "l", "usd_per_1m_input": 1, "usd_per_1m_output": 1}}}
    report, summary = evaluate_cascade(qs, collect, samples, cfg)
    casc = summary["stats"]["cascade (k=3, T=0.7)"]
    assert summary["n"] == 8 and summary["diagnostics"]["n_escalated"] == 2
    assert casc["accuracy"][0] == 1.0  # both small failures were caught and escalated
    assert "Primary pre-registered test" in report and "cascade" in report
    assert "cost-matched" in report and 0.0 <= summary["cost_matched_frac_large"] <= 1.0
