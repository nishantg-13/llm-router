import numpy as np
import pytest

from llm_router.policies import LearnedRouter, RulesRouter, default_policies


def test_rules_send_latex_and_long_prompts_to_large():
    r = RulesRouter(max_chars=100)
    assert r.choose("Let $\\frac{1}{2} + x = 3$. Find x.") == "large"
    assert r.choose("Prove that the integral converges.") == "large"
    assert r.choose("x" * 101) == "large"


def test_rules_keep_plain_word_problems_on_small():
    assert RulesRouter().choose("Tom has $5 and buys 3 apples at $1 each. How much is left?") == "small"


def test_default_policies_names():
    assert set(default_policies()) == {"always_small", "always_large", "rules"}


def fake_embed(texts):
    # one-dimensional feature: prompt length, so "short = easy" is learnable
    return np.array([[len(t) / 100.0] for t in texts])


def test_learned_router_learns_a_length_signal():
    prompts = ["a" * n for n in (10, 20, 30, 40, 200, 220, 240, 260)]
    small_ok = [True, True, True, True, False, False, False, False]
    r = LearnedRouter(fake_embed, threshold=0.5).fit(prompts, small_ok)
    assert r.choose("a" * 15) == "small"
    assert r.choose("a" * 250) == "large"


def test_higher_threshold_routes_more_to_large():
    prompts = ["a" * n for n in (10, 30, 60, 100, 140, 180, 220, 260)]
    small_ok = [True, True, True, True, False, True, False, False]
    r = LearnedRouter(fake_embed).fit(prompts, small_ok)
    probe = ["a" * n for n in range(10, 270, 20)]
    r.threshold = 0.5
    low = sum(r.choose(p) == "large" for p in probe)
    r.threshold = 0.95
    high = sum(r.choose(p) == "large" for p in probe)
    assert high >= low


def test_single_class_training_data_predicts_constant():
    r = LearnedRouter(fake_embed, threshold=0.9).fit(["a", "bb", "ccc"], [True, True, True])
    assert r.choose("anything") == "small"


def test_unfitted_router_raises():
    with pytest.raises(RuntimeError):
        LearnedRouter(fake_embed).p_small_ok(["x"])
