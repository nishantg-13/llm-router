import json

from fastapi.testclient import TestClient

from llm_router.gateway import create_app
from llm_router.policies import AlwaysLarge, AlwaysSmall
from tests.test_pool import FakeBackend, make_pool


def make_client(pool=None, cache_size=256):
    pool = pool or make_pool()
    policies = {p.name: p for p in (AlwaysSmall(), AlwaysLarge())}
    return TestClient(create_app(pool, policies, "always_small", cache_size)), pool


def test_complete_uses_default_policy_and_reports_meta():
    client, _ = make_client()
    body = client.post("/v1/complete", json={"prompt": "hi"}).json()
    assert (body["policy"], body["tier"], body["text"]) == ("always_small", "small", "hello")
    assert body["cost_usd"] > 0 and body["cached"] is False


def test_policy_override_routes_to_large():
    client, _ = make_client()
    body = client.post("/v1/complete", json={"prompt": "hi", "policy": "always_large"}).json()
    assert body["tier"] == "large"


def test_unknown_policy_is_400():
    client, _ = make_client()
    assert client.post("/v1/complete", json={"prompt": "hi", "policy": "nope"}).status_code == 400


def test_second_identical_request_is_served_from_cache_for_free():
    client, pool = make_client()
    client.post("/v1/complete", json={"prompt": "hi"})
    body = client.post("/v1/complete", json={"prompt": "hi"}).json()
    assert body["cached"] is True and body["cost_usd"] == 0.0
    assert pool.backends["small"].calls == 1


def test_cache_evicts_oldest_entry():
    client, pool = make_client(cache_size=1)
    client.post("/v1/complete", json={"prompt": "a"})
    client.post("/v1/complete", json={"prompt": "b"})
    client.post("/v1/complete", json={"prompt": "a"})
    assert pool.backends["small"].calls == 3


def test_fallback_answers_are_not_cached_under_the_chosen_tier():
    client, pool = make_client(make_pool(small=FakeBackend(fail_before=True)))
    first = client.post("/v1/complete", json={"prompt": "hi"}).json()
    assert first["tier"] == "large" and first["fell_back"] is True
    client.post("/v1/complete", json={"prompt": "hi"})
    assert pool.backends["small"].calls == 2  # tried the small tier again, nothing cached for it


def test_both_tiers_failing_is_502():
    pool = make_pool(small=FakeBackend(fail_before=True), large=FakeBackend(fail_before=True))
    client, _ = make_client(pool)
    assert client.post("/v1/complete", json={"prompt": "hi"}).status_code == 502


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        name, data = "message", None
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[7:]
            elif line.startswith("data: "):
                data = json.loads(line[6:])
        events.append((name, data))
    return events


def test_stream_emits_deltas_then_done():
    client, _ = make_client()
    events = parse_sse(client.post("/v1/stream", json={"prompt": "hi"}).text)
    assert [e[0] for e in events] == ["message", "message", "done"]
    assert "".join(e[1]["delta"] for e in events[:-1]) == "hello"
    assert events[-1][1]["tier"] == "small"


def test_stream_mid_failure_sends_error_event():
    client, _ = make_client(make_pool(small=FakeBackend(fail_after_first=True)))
    events = parse_sse(client.post("/v1/stream", json={"prompt": "hi"}).text)
    assert events[-1][0] == "error"


def test_metrics_aggregate_requests():
    client, _ = make_client()
    client.post("/v1/complete", json={"prompt": "a"})
    client.post("/v1/complete", json={"prompt": "a"})
    client.post("/v1/complete", json={"prompt": "b", "policy": "always_large"})
    m = client.get("/metrics").json()
    assert m["requests"] == 3 and m["cache_hits"] == 1
    assert m["by_tier"] == {"small": 2, "large": 1}
    assert m["latency_p50_s"] is not None
