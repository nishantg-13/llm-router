import asyncio

import pytest

from llm_router.config import ModelSpec
from llm_router.pool import Chunk, Pool

SPECS = {
    "small": ModelSpec("small-model", 1.0, 2.0),
    "large": ModelSpec("large-model", 10.0, 20.0),
}


class FakeBackend:
    def __init__(self, chunks=None, fail_before=False, fail_after_first=False, delay=0.0):
        self.chunks = chunks if chunks is not None else [Chunk("he"), Chunk("llo", 100, 50)]
        self.fail_before, self.fail_after_first, self.delay = fail_before, fail_after_first, delay
        self.calls = 0

    async def stream(self, prompt):
        self.calls += 1
        if self.fail_before:
            raise ConnectionError("boom")
        if self.delay:
            await asyncio.sleep(self.delay)
        for i, c in enumerate(self.chunks):
            if self.fail_after_first and i == 1:
                raise ConnectionError("mid-stream")
            yield c


def make_pool(small=None, large=None, timeout=5.0):
    return Pool({"small": small or FakeBackend(), "large": large or FakeBackend()}, SPECS, timeout)


async def test_generate_joins_text_and_prices_tokens():
    r = await make_pool().generate("q", "small")
    assert r.text == "hello"
    assert (r.tier, r.model, r.fell_back) == ("small", "small-model", False)
    assert (r.input_tokens, r.output_tokens) == (100, 50)
    assert r.cost_usd == pytest.approx((100 * 1.0 + 50 * 2.0) / 1_000_000)
    assert 0 <= r.ttft_s <= r.total_s


async def test_large_tier_costs_more_for_same_tokens():
    pool = make_pool()
    small, large = await pool.generate("q", "small"), await pool.generate("q", "large")
    assert large.cost_usd == pytest.approx(10 * small.cost_usd)


async def test_falls_back_when_chosen_tier_fails_before_first_token():
    pool = make_pool(small=FakeBackend(fail_before=True))
    r = await pool.generate("q", "small")
    assert (r.tier, r.fell_back, r.text) == ("large", True, "hello")


async def test_falls_back_on_timeout():
    pool = make_pool(small=FakeBackend(delay=0.5), timeout=0.05)
    r = await pool.generate("q", "small")
    assert (r.tier, r.fell_back) == ("large", True)


async def test_no_fallback_after_text_was_streamed():
    pool = make_pool(small=FakeBackend(fail_after_first=True))
    with pytest.raises(ConnectionError):
        await pool.generate("q", "small")
    assert pool.backends["large"].calls == 0


async def test_raises_when_both_tiers_fail():
    pool = make_pool(small=FakeBackend(fail_before=True), large=FakeBackend(fail_before=True))
    with pytest.raises(RuntimeError, match="both tiers failed"):
        await pool.generate("q", "small")


async def test_empty_stream_counts_as_failure_and_falls_back():
    pool = make_pool(small=FakeBackend(chunks=[]))
    r = await pool.generate("q", "small")
    assert (r.tier, r.fell_back) == ("large", True)


async def test_fallback_can_be_disabled_for_benchmarking():
    pool = make_pool(small=FakeBackend(fail_before=True))
    with pytest.raises(RuntimeError, match="small failed"):
        await pool.generate("q", "small", fallback=False)
    assert pool.backends["large"].calls == 0
