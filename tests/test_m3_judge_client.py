"""Judge clients: fallback chain, file cache and semaphore concurrency."""

import asyncio
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from rag_reliability.methods.m3.judge_client import (
    EXTRACTOR_VERSION,
    AsyncJudgeClient,
    JudgeClient,
    _cache_path,
    _choices_from_response,
)
from rag_reliability.methods.m3.logprobs import _pass_prob


def _choice(
    text: str,
    tokens: list[dict] | None = None,
    finish_reason: str = "stop",
) -> dict:
    return {"text": text, "tokens": tokens or [], "finish_reason": finish_reason}


def _judge(
    monkeypatch,
    reply: str,
    cache_dir=None,
    *,
    finish_reason: str = "stop",
) -> JudgeClient:
    client = JudgeClient(
        model="judge-model",
        api_base="https://example.test/v1",
        api_key="secret",
        cache_dir=cache_dir,
    )
    calls = []

    def fake(self, system, user, max_tokens):
        calls.append(user)
        return _choice(reply, finish_reason=finish_reason)

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
        {"token": "FAITHFULNESS", "logprob": -0.1, "top": {}},
        {"token": ":", "logprob": -0.1, "top": {}},
        {"token": " PASS", "logprob": -0.1, "top": {" PASS": -0.1, " FAIL": -2.4}},
        {"token": "\nRELEVANCE", "logprob": -0.1, "top": {}},
        {"token": ":", "logprob": -0.1, "top": {}},
        {"token": " FAIL", "logprob": -0.2, "top": {" FAIL": -0.2, " PASS": -1.7}},
    ]
    monkeypatch.setattr(
        JudgeClient,
        "_chat_judge",
        lambda self, s, u, mt: _choice(
            "FAITHFULNESS: PASS\nRELEVANCE: FAIL",
            tokens,
        ),
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
    out = asyncio.run(async_client.judge_many("sys", ["usr"]))
    assert out[0] == first


def test_semaphore_caps_concurrency(monkeypatch) -> None:
    client = AsyncJudgeClient(
        model="m",
        api_base="http://x",
        api_key="k",
        concurrency=2,
    )
    seen = {"active": 0, "max_active": 0, "calls": 0}

    async def fake(self, system, user, max_tokens):
        seen["active"] += 1
        seen["max_active"] = max(seen["max_active"], seen["active"])
        seen["calls"] += 1
        await asyncio.sleep(0.01)
        seen["active"] -= 1
        return _choice(f"FAITHFULNESS: PASS\nRELEVANCE: FAIL ({user[-6:]})")

    monkeypatch.setattr(AsyncJudgeClient, "_chat_judge_async", fake, raising=True)
    users = [f"user_{i:03d}" for i in range(6)]
    out = asyncio.run(client.judge_many("sys", users))
    assert len(out) == 6
    assert seen["calls"] == 6
    assert seen["max_active"] <= 2  # the semaphore works
    assert [o[2]["raw"][-4:-1] for o in out] == [f"{i:03d}" for i in range(6)]  # order kept


def test_cache_key_changes_with_extractor_version(tmp_path) -> None:
    common = {
        "cache_dir": tmp_path,
        "model": "model",
        "system": "system",
        "user": "user",
        "max_tokens": 800,
        "temperature": 0.0,
        "top_p": 1.0,
        "api_base": "https://example.test/v1",
    }
    old = _cache_path(**common, extractor_version=EXTRACTOR_VERSION - 1)
    current = _cache_path(**common, extractor_version=EXTRACTOR_VERSION)
    assert old != current


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("max_tokens", 799),
        ("temperature", 0.2),
        ("top_p", 0.9),
        ("api_base", "https://other.test/v1"),
    ],
)
def test_cache_key_includes_generation_parameters(tmp_path, field: str, replacement) -> None:
    common = {
        "cache_dir": tmp_path,
        "model": "model",
        "system": "system",
        "user": "user",
        "max_tokens": 800,
        "temperature": 0.0,
        "top_p": 1.0,
        "api_base": "https://example.test/v1",
        "extractor_version": EXTRACTOR_VERSION,
    }
    original = _cache_path(**common)
    changed = _cache_path(**(common | {field: replacement}))
    assert original != changed


def test_cache_stores_raw_tokens_not_probabilities(monkeypatch, tmp_path) -> None:
    tokens = [
        {"token": "FAITHFULNESS", "logprob": -0.1, "top": {}},
        {"token": ":", "logprob": -0.1, "top": {}},
        {"token": " PASS", "logprob": -0.1, "top": {" PASS": -0.1}},
        {"token": "\nRELEVANCE", "logprob": -0.1, "top": {}},
        {"token": ":", "logprob": -0.1, "top": {}},
        {"token": " FAIL", "logprob": -0.2, "top": {" FAIL": -0.2}},
    ]
    client = _judge(
        monkeypatch,
        "FAITHFULNESS: PASS\nRELEVANCE: FAIL",
        cache_dir=tmp_path,
    )
    monkeypatch.setattr(
        JudgeClient,
        "_chat_judge",
        lambda self, s, u, mt: _choice(
            "FAITHFULNESS: PASS\nRELEVANCE: FAIL",
            tokens,
        ),
        raising=True,
    )
    client.judge("sys", "usr")

    cache_files = list(tmp_path.glob("*.json"))
    assert len(cache_files) == 1
    payload = json.loads(cache_files[0].read_text(encoding="utf-8"))
    assert payload == {
        "text": "FAITHFULNESS: PASS\nRELEVANCE: FAIL",
        "tokens": tokens,
        "finish_reason": "stop",
    }
    assert "p_faith" not in payload
    assert "p_rel" not in payload


def test_finish_reason_length_sets_truncated(monkeypatch) -> None:
    client = _judge(
        monkeypatch,
        "FAITHFULNESS: PASS\nRELEVANCE: FAIL",
        finish_reason="length",
    )
    _, _, meta = client.judge("sys", "usr")
    assert meta["truncated"] is True


def test_default_judge_budget_is_800(monkeypatch) -> None:
    client = _judge(monkeypatch, "FAITHFULNESS: PASS\nRELEVANCE: PASS")
    seen: list[int] = []

    def fake(self, system, user, max_tokens):
        seen.append(max_tokens)
        return _choice("FAITHFULNESS: PASS\nRELEVANCE: PASS")

    monkeypatch.setattr(JudgeClient, "_chat_judge", fake, raising=True)
    client.judge("sys", "usr")
    assert seen == [800]


def test_logprob_http_error_degrades_to_text_and_logs(monkeypatch, caplog) -> None:
    client = JudgeClient(
        model="judge-model",
        api_base="https://example.test/v1",
        api_key="secret",
    )

    def fake_chat(messages, **kwargs):
        if kwargs["logprobs"]:
            raise ValueError("HTTP 400: logprobs unsupported")
        return [_choice("FAITHFULNESS: PASS\nRELEVANCE: FAIL")]

    monkeypatch.setattr(client, "chat", fake_chat)
    choice = client._chat_judge("sys", "usr", 800)
    assert choice["tokens"] == []
    assert "HTTP 400: logprobs unsupported" in caplog.text


def test_choice_conversion_keeps_finish_reason() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="text"),
                logprobs=None,
                finish_reason="length",
            )
        ]
    )
    assert _choices_from_response(response, logprobs=False) == [
        {"text": "text", "tokens": [], "finish_reason": "length"}
    ]


def test_recomputation_fixes_known_mismatches() -> None:
    artifact = (
        Path(__file__).parents[1]
        / "predictions"
        / "pseudo_debug"
        / "m3"
        / "gepa_plain_s1"
        / "val.jsonl"
    )
    known_mismatch_ids = {
        "pseudo_00017",
        "pseudo_00025",
        "pseudo_00058",
        "pseudo_00063",
        "pseudo_00074",
        "pseudo_00097",
        "pseudo_00135",
        "pseudo_00150",
        "pseudo_00197",
        "pseudo_00265",
        "pseudo_00287",
    }
    rows = {
        row["id"]: row
        for row in (
            json.loads(line) for line in artifact.read_text(encoding="utf-8").splitlines()
        )
        if row["id"] in known_mismatch_ids
    }

    original_mismatches = 0
    recomputed_mismatches = 0
    for row in rows.values():
        expected = (1, 1)
        original = (int(row["p_faith"] >= 0.5), int(row["p_rel"] >= 0.5))
        recomputed_probs = []
        for old_probability in (row["p_faith"], row["p_rel"]):
            if old_probability >= 0.5:
                recomputed_probs.append(old_probability)
                continue
            historical_logprob = math.log(old_probability / (1.0 - old_probability))
            recomputed_probs.append(
                _pass_prob(
                    {
                        "token": " PASS",
                        "logprob": historical_logprob,
                        "top": {" PASS": historical_logprob},
                    }
                )
            )
        recomputed = tuple(int(probability >= 0.5) for probability in recomputed_probs)
        original_mismatches += original != expected
        recomputed_mismatches += recomputed != expected

    assert set(rows) == known_mismatch_ids
    assert original_mismatches == 11
    assert recomputed_mismatches == 0
