"""Judge clients: fallback chain, file cache, semaphore concurrency and guard."""

import asyncio

import pytest

from rag_reliability.guard import DataLeakError
from rag_reliability.methods.m3.judge_client import AsyncJudgeClient, JudgeClient
from rag_reliability.schema import RagSample


def _sample(id: str = "pseudo_1") -> RagSample:
    return RagSample(id=id, question="q", context="c", answer="a", faithfulness=1, relevance=1)


def _judge(monkeypatch, reply: str, cache_dir=None) -> JudgeClient:
    client = JudgeClient(
        model="judge-model",
        api_base="https://example.test/v1",
        api_key="secret",
        cache_dir=cache_dir,
    )
    calls = []

    def fake(self, system, user, max_tokens, sample=None):
        calls.append(user)
        return reply, []

    monkeypatch.setattr(JudgeClient, "_chat_judge", fake, raising=True)
    client._test_calls = calls  # type: ignore[attr-defined]
    return client


def test_regex_fallback(monkeypatch) -> None:
    client = _judge(monkeypatch, "Анализ...\nFAITHFULNESS: PASS\nRELEVANCE: FAIL")
    p_f, p_r, meta = client.judge("sys", "usr")
    assert (p_f, p_r) == (0.9, 0.1)
    assert meta["method"] == "regex"


def test_unparseable_gives_half(monkeypatch) -> None:
    client = _judge(monkeypatch, "бессвязный текст")
    p_f, p_r, meta = client.judge("sys", "usr")
    assert (p_f, p_r) == (0.5, 0.5)
    assert meta["method"] == "default"


def test_logprobs_path_preferred(monkeypatch) -> None:
    client = JudgeClient(
        model="judge-model",
        api_base="https://example.test/v1",
        api_key="secret",
    )
    tokens = [
        {"token": " PASS", "logprob": -0.1, "top": {" PASS": -0.1, " FAIL": -2.4}},
        {"token": " FAIL", "logprob": -0.2, "top": {" FAIL": -0.2, " PASS": -1.7}},
    ]
    monkeypatch.setattr(
        JudgeClient,
        "_chat_judge",
        lambda self, s, u, mt, sample=None: ("FAITHFULNESS: PASS\nRELEVANCE: FAIL", tokens),
        raising=True,
    )
    p_f, p_r, meta = client.judge("sys", "usr")
    assert meta["method"] == "logprobs"
    assert 0.5 < p_f < 1.0
    assert 0.0 < p_r < 0.5


def test_cache_reused_across_sync_and_async(monkeypatch, tmp_path) -> None:
    client = _judge(monkeypatch, "FAITHFULNESS: PASS\nRELEVANCE: PASS", cache_dir=tmp_path)
    first = client.judge("sys", "usr")
    second = client.judge("sys", "usr")
    assert first == second
    assert len(client._test_calls) == 1  # type: ignore[attr-defined]

    # The async judge hits the same cache entry without any transport call.
    async_client = AsyncJudgeClient(
        model="judge-model",
        api_base="https://example.test/v1",
        api_key="secret",
        cache_dir=tmp_path,
    )

    async def boom(self, system, user, max_tokens):
        raise AssertionError("cache miss: async transport must not be called")

    monkeypatch.setattr(AsyncJudgeClient, "_chat_judge_async", boom, raising=True)
    out = asyncio.run(async_client.judge_many("sys", [(_sample(), "usr")]))
    assert out[0] == first


def _mk_async_client(monkeypatch, delay: float = 0.01):
    client = AsyncJudgeClient(
        model="m",
        api_base="http://x",
        api_key="k",
        profile="cloud",
        concurrency=2,
    )
    seen = {"active": 0, "max_active": 0, "calls": 0}

    async def fake(self, system, user, max_tokens):
        seen["active"] += 1
        seen["max_active"] = max(seen["max_active"], seen["active"])
        seen["calls"] += 1
        await asyncio.sleep(delay)
        seen["active"] -= 1
        return f"FAITHFULNESS: PASS\nRELEVANCE: FAIL ({user[-6:]})", []

    monkeypatch.setattr(AsyncJudgeClient, "_chat_judge_async", fake, raising=True)
    return client, seen


def test_semaphore_caps_concurrency(monkeypatch) -> None:
    client, seen = _mk_async_client(monkeypatch)
    items = [(_sample(id=f"pseudo_{i}"), f"user_{i:03d}") for i in range(6)]
    out = asyncio.run(client.judge_many("sys", items))
    assert len(out) == 6
    assert seen["calls"] == 6
    assert seen["max_active"] <= 2  # the semaphore works
    assert [o[2]["raw"][-4:-1] for o in out] == [f"{i:03d}" for i in range(6)]  # order kept


def test_guard_fires_before_any_call(monkeypatch) -> None:
    client, seen = _mk_async_client(monkeypatch)
    bad = _sample(id="case_1")
    with pytest.raises(DataLeakError):
        asyncio.run(client.judge_many("sys", [(bad, "u")]))
    assert seen["calls"] == 0


def test_sync_chat_requires_marking_in_cloud() -> None:
    client = JudgeClient(
        model="m",
        api_base="http://x",
        api_key="k",
        profile="cloud",
    )
    with pytest.raises(DataLeakError):
        client.chat([{"role": "user", "content": "hi"}])
