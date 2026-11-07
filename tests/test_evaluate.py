import numpy as np
import pytest

from llm_router.bench.data import Question
from llm_router.bench.evaluate import Arrays, apply_choice, bootstrap, paired_diff, split_ids


def arrays(correct_small, correct_large, cost_small=1.0, cost_large=10.0):
    n = len(correct_small)
    return Arrays(
        correct=np.array([correct_small, correct_large]),
        cost=np.array([[cost_small] * n, [cost_large] * n]),
        total=np.array([[1.0] * n, [4.0] * n]),
        ttft=np.array([[0.5] * n, [2.0] * n]),
    )


def test_apply_choice_hard_policy_picks_recorded_tier():
    a = arrays([1, 0, 1, 0], [1, 1, 1, 1])
    out = apply_choice(a, np.array([0, 1, 0, 1]))
    assert out["correct"].tolist() == [1, 1, 1, 1]
    assert out["cost"].tolist() == [1.0, 10.0, 1.0, 10.0]
    assert out["total"].tolist() == [1.0, 4.0, 1.0, 4.0]


def test_apply_choice_fractional_gives_expectation_and_no_latency():
    a = arrays([1, 0], [1, 1])
    out = apply_choice(a, np.array([0.5, 0.5]))
    assert out["correct"].tolist() == [1.0, 0.5]
    assert out["cost"].tolist() == [5.5, 5.5]
    assert np.isnan(out["total"]).all()


def test_bootstrap_reference_vs_itself_is_exactly_zero_and_100_percent():
    a = arrays([1, 0, 1, 1, 0, 1], [1, 1, 1, 1, 1, 0])
    s = bootstrap(a, {"always_large": np.ones(6), "always_small": np.zeros(6)}, "always_large", B=200)
    assert s["always_large"]["acc_delta_vs_ref"] == (0.0, 0.0, 0.0)
    assert s["always_large"]["cost_pct_of_ref"][0] == pytest.approx(100.0)
    assert s["always_small"]["cost_pct_of_ref"][0] == pytest.approx(10.0)


def test_bootstrap_interval_brackets_point_estimate_and_orders_latency():
    rng = np.random.default_rng(1)
    small, large = rng.integers(0, 2, 60).tolist(), [1] * 60
    s = bootstrap(arrays(small, large), {"always_large": np.ones(60), "always_small": np.zeros(60)}, "always_large", B=300)
    v, lo, hi = s["always_small"]["accuracy"]
    assert lo <= v <= hi
    assert s["always_small"]["total_p50"][0] < s["always_large"]["total_p50"][0]


def test_oracle_matches_large_accuracy_when_large_is_perfect_and_costs_less():
    small = [1, 0, 1, 0, 1, 1, 0, 1]
    a = arrays(small, [1] * 8)
    oracle = (a.correct[0] == 0).astype(float)
    s = bootstrap(a, {"always_large": np.ones(8), "oracle": oracle}, "always_large", B=100)
    assert s["oracle"]["accuracy"][0] == 1.0
    assert s["oracle"]["cost_pct_of_ref"][0] < 100.0


def test_paired_diff_zero_for_identical_policies():
    a = arrays([1, 0, 1, 0], [1, 1, 1, 1])
    p = np.array([0.0, 1.0, 0.0, 1.0])
    assert paired_diff(a, p, p, B=100)[0] == 0.0


def _qs():
    return [Question(f"{s}-{i}", s, "x", "num", "p", "1") for s in ("gsm8k", "mmlu") for i in range(10)]


def test_split_is_stratified_disjoint_and_deterministic():
    train, test = split_ids(_qs(), 0.5, 7)
    assert {q.id for q in train}.isdisjoint({q.id for q in test})
    assert len(train) == len(test) == 10
    assert sum(q.source == "gsm8k" for q in train) == 5
    assert [q.id for q in split_ids(_qs(), 0.5, 7)[0]] == [q.id for q in train]


def test_auc_perfect_and_inverted_and_single_class():
    from llm_router.bench.evaluate import auc_with_ci

    y = np.array([1, 1, 1, 0, 0, 0] * 5)
    perfect = np.array([0.9, 0.8, 0.95, 0.1, 0.2, 0.05] * 5)
    assert auc_with_ci(perfect, y, B=100)[0] == 1.0
    assert auc_with_ci(1 - perfect, y, B=100)[0] == 0.0
    assert auc_with_ci(perfect, np.ones(30), B=100) is None


def test_ranked_frontier_perfect_ranker_beats_random_and_matches_oracle():
    from llm_router.bench.evaluate import ranked_frontier

    small = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    a = arrays(small, [1] * 10)
    p_ok = np.array([0.9] * 8 + [0.1, 0.1])  # perfectly ranks the two failures lowest
    rows = ranked_frontier(a, p_ok, (0.0, 0.2, 1.0), seed=1)
    assert rows[0]["learned_acc"][0] == pytest.approx(0.8)  # nothing routed: small-model accuracy
    assert rows[1]["learned_acc"][0] == pytest.approx(1.0)  # the two failures went to large
    assert rows[1]["random_acc"][0] == pytest.approx(0.84)
    assert rows[1]["diff_vs_random"][0] == pytest.approx(0.16)
    assert rows[2]["learned_acc"][0] == pytest.approx(1.0)
