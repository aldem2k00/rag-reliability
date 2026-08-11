"""Пофрагментная верификация Метода 3 на dummy batch-бэкенде."""

from __future__ import annotations

import asyncio
import math
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from rag_reliability.methods.m3.perchunk import score_per_chunk
from rag_reliability.schema import RagSample


class DummyBatchJudge:
    """Детерминированный batch-бэкенд без модели и сети."""

    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = list(scores)
        self.network_calls = 0
        self.systems: list[str] = []
        self.users: list[str] = []

    def __call__(self, system: str, users: Sequence[str]) -> list[float]:
        self.network_calls += 1
        self.systems.append(system)
        self.users.extend(users)
        if len(users) != len(self.scores):
            raise AssertionError(
                f"Dummy got {len(users)} prompt(s), but has {len(self.scores)} score(s)"
            )
        return list(self.scores)


class AsyncDummyBatchJudge:
    """Форма адаптера для AsyncJudgeClient: один await на весь кейс."""

    def __init__(self, scores: Sequence[float]) -> None:
        self.scores = list(scores)
        self.network_calls = 0

    async def __call__(self, system: str, users: Sequence[str]) -> list[float]:
        self.network_calls += 1
        assert "FAITHFULNESS" in system
        assert len(users) == len(self.scores)
        return list(self.scores)


class DummyAsyncChatClient:
    """Raw-choice клиент в форме ``AsyncJudgeClient.chat``."""

    def __init__(self, cache_dir: Path | None = None) -> None:
        self.concurrency = 3
        self.cache_dir = cache_dir
        self.model = "dummy-model"
        self.api_base = "dummy://local"
        self.calls = 0
        self.active = 0
        self.max_active = 0

    async def chat(
        self,
        messages: list[dict[str, str]],
        **kwargs: Any,
    ) -> list[dict]:
        self.calls += 1
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        await asyncio.sleep(0)
        self.active -= 1

        user = messages[-1]["content"]
        match = re.search(r"УНИКАЛЬНЫЙ_ЧАНК_(\d+)", user)
        assert match is not None
        index = int(match.group(1))
        if index == 1:
            probability = 0.8
            return [
                {
                    "text": "ANALYSIS: опора есть\nMARKER: none\nFAITHFULNESS: PASS",
                    "tokens": [
                        {"token": "FAITHFULNESS", "logprob": -0.1, "top": {}},
                        {"token": ":", "logprob": -0.1, "top": {}},
                        {
                            "token": " PASS",
                            "logprob": math.log(probability),
                            "top": {
                                " PASS": math.log(probability),
                                " FAIL": math.log(1.0 - probability),
                            },
                        },
                    ],
                    "finish_reason": "stop",
                }
            ]
        if index == 2:
            return [
                {
                    "text": "ANALYSIS: опоры нет\nMARKER: none\nFAITHFULNESS: FAIL",
                    "tokens": [],
                    "finish_reason": "stop",
                }
            ]
        return [{"text": "невалидный ответ", "tokens": [], "finish_reason": "stop"}]


def make_sample(n_chunks: int) -> RagSample:
    context = "\n\n".join(
        f"[CHUNK {index}]\nУНИКАЛЬНЫЙ_ЧАНК_{index}" for index in range(1, n_chunks + 1)
    )
    return RagSample(
        id=f"case-{n_chunks}",
        question="Как выполнить операцию?",
        context=context,
        answer="Выполните указанные шаги.",
        faithfulness=1,
        relevance=1,
        marker="none",
    )


@pytest.mark.parametrize("n_chunks", [5, 8])
def test_one_prompt_per_chunk_is_sent_in_one_batch(n_chunks: int) -> None:
    judge = DummyBatchJudge([0.6] * n_chunks)

    score_per_chunk(make_sample(n_chunks), judge)

    assert len(judge.users) == n_chunks
    assert judge.network_calls == 1


def test_each_prompt_contains_only_its_own_chunk() -> None:
    judge = DummyBatchJudge([0.2, 0.5, 0.8])

    score_per_chunk(make_sample(3), judge)

    for index, user in enumerate(judge.users, start=1):
        assert f"УНИКАЛЬНЫЙ_ЧАНК_{index}" in user
        other_chunks = {
            f"УНИКАЛЬНЫЙ_ЧАНК_{other}"
            for other in range(1, 4)
            if other != index
        }
        assert all(chunk not in user for chunk in other_chunks)


def test_five_features_are_aggregated_from_chunk_scores() -> None:
    judge = DummyBatchJudge([0.95, 0.8, 0.2, 0.1])

    scores = score_per_chunk(make_sample(4), judge)

    assert set(scores) == {
        "m3.max_chunk_score",
        "m3.mean_chunk_score",
        "m3.chunk_disagreement",
        "m3.n_supporting",
        "m3.argmax_chunk",
    }
    assert scores["m3.max_chunk_score"] == pytest.approx(0.95)
    assert scores["m3.mean_chunk_score"] == pytest.approx(0.5125)
    assert scores["m3.chunk_disagreement"] == pytest.approx(0.15)
    assert scores["m3.n_supporting"] == 2.0
    assert scores["m3.argmax_chunk"] == 1.0


def test_chunk_disagreement_is_zero_for_equal_scores_and_maximal_for_one_peak() -> None:
    equal = score_per_chunk(make_sample(4), DummyBatchJudge([0.7] * 4))
    one_peak = score_per_chunk(make_sample(4), DummyBatchJudge([1.0, 0.0, 0.0, 0.0]))

    assert equal["m3.chunk_disagreement"] == 0.0
    assert one_peak["m3.chunk_disagreement"] == 1.0


def test_argmax_chunk_is_one_based_and_matches_the_best_source() -> None:
    scores = score_per_chunk(make_sample(4), DummyBatchJudge([0.1, 0.2, 0.9, 0.4]))

    assert scores["m3.argmax_chunk"] == 3.0


def test_single_chunk_has_zero_disagreement() -> None:
    scores = score_per_chunk(make_sample(1), DummyBatchJudge([0.73]))

    assert scores["m3.max_chunk_score"] == pytest.approx(0.73)
    assert scores["m3.chunk_disagreement"] == 0.0
    assert scores["m3.argmax_chunk"] == 1.0


def test_async_batch_backend_is_awaited_once() -> None:
    judge = AsyncDummyBatchJudge([0.2, 0.9, 0.4])

    scores = score_per_chunk(make_sample(3), judge)

    assert judge.network_calls == 1
    assert scores["m3.argmax_chunk"] == 2.0


def test_async_judge_client_shape_batches_and_uses_axis_fallback_chain() -> None:
    client = DummyAsyncChatClient()

    scores = score_per_chunk(make_sample(3), client)

    assert client.calls == 3
    assert client.max_active == 3
    assert scores["m3.max_chunk_score"] == pytest.approx(0.8)
    assert scores["m3.mean_chunk_score"] == pytest.approx((0.8 + 0.1 + 0.5) / 3)
    assert scores["m3.argmax_chunk"] == 1.0


def test_async_judge_client_raw_choices_are_cached(tmp_path: Path) -> None:
    client = DummyAsyncChatClient(cache_dir=tmp_path)
    sample = make_sample(3)

    first = score_per_chunk(sample, client)
    second = score_per_chunk(sample, client)

    assert second == first
    assert client.calls == 3


def test_relevance_is_rejected_before_the_backend_is_called() -> None:
    judge = DummyBatchJudge([0.5])

    with pytest.raises(ValueError, match="only axis='faithfulness'"):
        score_per_chunk(make_sample(1), judge, axis="relevance")

    assert judge.network_calls == 0


def test_empty_context_fails_loudly() -> None:
    judge = DummyBatchJudge([])
    sample = make_sample(1).model_copy(update={"context": "   "})

    with pytest.raises(ValueError, match="has no context chunks"):
        score_per_chunk(sample, judge)

    assert judge.network_calls == 0


@pytest.mark.parametrize(
    ("backend_scores", "message"),
    [
        ([0.2], "expected one score for each of 2 chunk"),
        ([0.2, 1.1], "invalid probability"),
        ([0.2, float("nan")], "invalid probability"),
    ],
)
def test_backend_scores_are_complete_probabilities(
    backend_scores: list[float], message: str
) -> None:
    judge = DummyBatchJudge(backend_scores)
    sample = make_sample(2)

    if len(backend_scores) != 2:
        # Этот dummy обычно проверяет выравнивание сам; здесь нужен ответ
        # неправильной длины от внешнего batch-бэкенда.
        judge.scores = [0.2, 0.3]

        def incomplete(system: str, users: Sequence[str]) -> list[float]:
            judge.network_calls += 1
            return backend_scores

        backend = incomplete
    else:
        backend = judge

    with pytest.raises(ValueError, match=message):
        score_per_chunk(sample, backend)
