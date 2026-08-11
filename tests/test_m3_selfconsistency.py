"""Self-consistency и абляция качество/цена. Всё на подставном клиенте, без сети."""

from __future__ import annotations

import pytest

from rag_reliability.methods.m3.axes import AXIS_FAITHFULNESS, AXIS_RELEVANCE
from rag_reliability.methods.m3.selfconsistency import (
    AblationRecord,
    aggregate_axis,
    aggregate_prefix,
    build_ablation_table,
    judge_selfconsistent,
    render_ablation_markdown,
    sample_axis,
)


class FakeClient:
    """Клиент судьи с заранее заданными ответами; повторяет контракт chat()."""

    def __init__(self, choices: list[dict], *, logprobs_supported: bool = True) -> None:
        self._choices = choices
        self.logprobs_supported = logprobs_supported
        self.calls: list[dict] = []

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float = 0.0,
        n: int = 1,
        max_tokens: int = 512,
        top_p: float = 1.0,
        logprobs: bool = False,
    ) -> list[dict]:
        if logprobs and not self.logprobs_supported:
            raise RuntimeError("HTTP 400: logprobs unsupported")
        self.calls.append(
            {"n": n, "temperature": temperature, "logprobs": logprobs, "messages": messages}
        )
        if n > len(self._choices):
            raise AssertionError(f"fake client has {len(self._choices)} choices, asked for {n}")
        return self._choices[:n]


def _choice(axis: str, verdict: str, p_logprob: float, f_logprob: float) -> dict:
    anchor = axis.upper()
    text = f"ANALYSIS: разбор\nMARKER: none\n{anchor}: {verdict}"
    return {
        "text": text,
        "tokens": [
            {"token": f"{anchor}", "logprob": -0.1, "top": {}},
            {"token": ":", "logprob": -0.1, "top": {}},
            {
                "token": f" {verdict}",
                "logprob": p_logprob,
                "top": {" PASS": p_logprob, " FAIL": f_logprob},
            },
        ],
        "finish_reason": "stop",
    }


def _text_choice(axis: str, verdict: str) -> dict:
    return {
        "text": f"ANALYSIS: разбор\nMARKER: none\n{axis.upper()}: {verdict}",
        "tokens": [],
        "finish_reason": "stop",
    }


def test_n1_matches_a_single_call() -> None:
    choice = _choice(AXIS_RELEVANCE, "PASS", -0.05, -3.0)
    client = FakeClient([choice])

    result = judge_selfconsistent(client, "sys", "usr", axis=AXIS_RELEVANCE, n=1)

    assert client.calls[0]["n"] == 1
    assert result[f"{AXIS_RELEVANCE}.p"] == pytest.approx(result["meta"]["probs"][0])
    assert result[f"{AXIS_RELEVANCE}.p_std"] == 0.0
    assert result[f"{AXIS_RELEVANCE}.p_vote"] == 1.0


def test_probabilities_are_averaged_not_votes() -> None:
    """Голоса дают гранулярность 1/N; вероятности непрерывны — усредняем их."""
    choices = [
        _choice(AXIS_FAITHFULNESS, "PASS", -0.0001, -9.0),  # p ~ 1.0
        _choice(AXIS_FAITHFULNESS, "PASS", -0.6931, -0.6931),  # p = 0.5
    ]
    client = FakeClient(choices)

    result = judge_selfconsistent(client, "sys", "usr", axis=AXIS_FAITHFULNESS, n=2)

    probs = result["meta"]["probs"]
    assert result[f"{AXIS_FAITHFULNESS}.p"] == pytest.approx(sum(probs) / 2)
    # Оба сэмпла проголосовали PASS, но средняя вероятность заметно ниже 1.0.
    assert result[f"{AXIS_FAITHFULNESS}.p_vote"] == 1.0
    assert result[f"{AXIS_FAITHFULNESS}.p"] < 0.95


def test_std_is_zero_for_identical_samples_and_positive_otherwise() -> None:
    identical = FakeClient([_choice(AXIS_RELEVANCE, "PASS", -0.1, -2.0)] * 4)
    mixed = FakeClient(
        [
            _choice(AXIS_RELEVANCE, "PASS", -0.02, -4.0),
            _choice(AXIS_RELEVANCE, "FAIL", -4.0, -0.02),
            _choice(AXIS_RELEVANCE, "PASS", -0.5, -0.9),
            _choice(AXIS_RELEVANCE, "FAIL", -2.0, -0.2),
        ]
    )

    same = judge_selfconsistent(identical, "sys", "usr", axis=AXIS_RELEVANCE, n=4)
    different = judge_selfconsistent(mixed, "sys", "usr", axis=AXIS_RELEVANCE, n=4)

    assert same[f"{AXIS_RELEVANCE}.p_std"] == 0.0
    assert different[f"{AXIS_RELEVANCE}.p_std"] > 0.0
    assert different[f"{AXIS_RELEVANCE}.p_vote"] == 0.5


def test_temperature_and_n_reach_the_client() -> None:
    client = FakeClient([_choice(AXIS_RELEVANCE, "PASS", -0.1, -2.0)] * 8)

    judge_selfconsistent(client, "sys", "usr", axis=AXIS_RELEVANCE, n=8, temperature=0.7)

    assert client.calls[0]["n"] == 8
    assert client.calls[0]["temperature"] == 0.7
    assert client.calls[0]["logprobs"] is True


def test_provider_without_logprobs_degrades_to_text() -> None:
    client = FakeClient([_text_choice(AXIS_RELEVANCE, "FAIL")] * 2, logprobs_supported=False)

    result = judge_selfconsistent(client, "sys", "usr", axis=AXIS_RELEVANCE, n=2)

    assert result["meta"]["methods"] == ["regex", "regex"]
    assert result[f"{AXIS_RELEVANCE}.p"] == pytest.approx(0.1)
    assert result[f"{AXIS_RELEVANCE}.p_vote"] == 0.0


def test_missing_samples_are_not_silently_dropped() -> None:
    class ShortClient(FakeClient):
        def chat(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            super().chat(messages, **kwargs)
            return self._choices[:1]

    client = ShortClient([_choice(AXIS_RELEVANCE, "PASS", -0.1, -2.0)] * 4)

    with pytest.raises(RuntimeError, match="silently dropped"):
        judge_selfconsistent(client, "sys", "usr", axis=AXIS_RELEVANCE, n=4)


def test_samples_are_cached_per_axis_and_reused(tmp_path) -> None:
    """Сэмплирование судьи — самая дорогая операция метода; перезапуск должен
    продолжать, а не пересчитывать."""
    choices = [_choice(AXIS_RELEVANCE, "PASS", -0.1, -2.0)] * 4
    first = FakeClient(choices)
    second = FakeClient(choices)

    warm = judge_selfconsistent(
        first, "sys", "usr", axis=AXIS_RELEVANCE, n=4, cache_dir=tmp_path, cache_scope="model|url"
    )
    hit = judge_selfconsistent(
        second, "sys", "usr", axis=AXIS_RELEVANCE, n=4, cache_dir=tmp_path, cache_scope="model|url"
    )

    assert len(first.calls) == 1
    assert second.calls == []  # второй прогон не ходил в модель
    assert hit[f"{AXIS_RELEVANCE}.p"] == warm[f"{AXIS_RELEVANCE}.p"]
    assert len(list(tmp_path.glob("*.json"))) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [("axis", AXIS_FAITHFULNESS), ("n", 2), ("temperature", 1.0), ("cache_scope", "other")],
)
def test_cache_key_separates_runs_that_are_not_interchangeable(tmp_path, field, value) -> None:
    base = {
        "axis": AXIS_RELEVANCE,
        "n": 4,
        "temperature": 0.7,
        "cache_dir": tmp_path,
        "cache_scope": "model|url",
    }
    choices = [_choice(AXIS_RELEVANCE, "PASS", -0.1, -2.0)] * 4

    judge_selfconsistent(FakeClient(choices), "sys", "usr", **base)
    second = FakeClient([_choice(value if field == "axis" else AXIS_RELEVANCE, "PASS", -0.1, -2.0)] * 4)
    judge_selfconsistent(second, "sys", "usr", **(base | {field: value}))

    assert len(second.calls) == 1  # ключи различаются, кэш не переиспользован
    assert len(list(tmp_path.glob("*.json"))) == 2


def test_corrupted_cache_entry_counts_as_a_miss(tmp_path) -> None:
    choices = [_choice(AXIS_RELEVANCE, "PASS", -0.1, -2.0)] * 2
    judge_selfconsistent(
        FakeClient(choices), "sys", "usr", axis=AXIS_RELEVANCE, n=2, cache_dir=tmp_path
    )
    entry = next(iter(tmp_path.glob("*.json")))
    entry.write_text("{ oborvano", encoding="utf-8")

    client = FakeClient(choices)
    judge_selfconsistent(client, "sys", "usr", axis=AXIS_RELEVANCE, n=2, cache_dir=tmp_path)

    assert len(client.calls) == 1


def test_sample_axis_rejects_non_positive_n() -> None:
    with pytest.raises(ValueError, match="n must be >= 1"):
        sample_axis(FakeClient([]), "sys", "usr", axis=AXIS_RELEVANCE, n=0)


def test_aggregate_axis_requires_aligned_inputs() -> None:
    with pytest.raises(ValueError, match="aligned per sample"):
        aggregate_axis(AXIS_RELEVANCE, [0.5, 0.6], [1])
    with pytest.raises(ValueError, match="zero samples"):
        aggregate_axis(AXIS_RELEVANCE, [], [])


def test_unrecoverable_verdicts_fall_back_to_half_vote() -> None:
    aggregate = aggregate_axis(AXIS_RELEVANCE, [0.5, 0.5], [None, None])

    assert aggregate.p_vote == 0.5
    assert aggregate.n_voted == 0


# --- абляция ----------------------------------------------------------------


def _records() -> list[AblationRecord]:
    """Синтетика с настоящим сигналом: у надёжных кейсов вероятности выше."""
    records = []
    for index in range(12):
        gold = int(index % 2 == 0)
        base = 0.8 if gold else 0.4
        probs = tuple(base + 0.01 * step - 0.004 * index for step in range(4))
        votes = tuple(int(value >= 0.5) for value in probs)
        records.append(
            AblationRecord(case_id=f"c{index}", probs=probs, votes=votes, gold=gold)
        )
    return records


def test_aggregate_prefix_uses_only_the_first_n_samples() -> None:
    probs = (0.1, 0.9, 0.9, 0.9)
    votes = (0, 1, 1, 1)

    assert aggregate_prefix(probs, votes, 1, "mean_prob") == pytest.approx(0.1)
    assert aggregate_prefix(probs, votes, 4, "mean_prob") == pytest.approx(0.7)
    assert aggregate_prefix(probs, votes, 4, "vote_share") == pytest.approx(0.75)
    assert aggregate_prefix(probs, votes, 4, "median_prob") == pytest.approx(0.9)


def test_aggregate_prefix_rejects_unknown_aggregation_and_too_large_n() -> None:
    with pytest.raises(ValueError, match="Unknown aggregation"):
        aggregate_prefix((0.5,), (1,), 1, "majority")
    with pytest.raises(ValueError, match="only 1 sample"):
        aggregate_prefix((0.5,), (1,), 2, "mean_prob")


def test_ablation_table_covers_the_whole_grid_with_cost_and_ci() -> None:
    rows = build_ablation_table(
        _records(),
        axis=AXIS_FAITHFULNESS,
        temperature=0.7,
        ns=[1, 2, 4],
        replicates=200,
    )

    assert len(rows) == 3 * 3  # N x агрегация
    for row in rows:
        assert row["ci95"][0] <= row["auc"] <= row["ci95"][1]
        assert row["cost_samples_per_case"] == row["n"]
        assert row["cost_relative"] == row["n"]
        assert 0.0 <= row["pass_rate"] <= 1.0
    assert {row["n"] for row in rows} == {1, 2, 4}


def test_ablation_table_refuses_cases_with_too_few_samples() -> None:
    records = [
        AblationRecord(case_id="short", probs=(0.5,), votes=(1,), gold=1),
        AblationRecord(case_id="ok", probs=(0.5, 0.6), votes=(1, 1), gold=0),
    ]

    with pytest.raises(ValueError, match="fewer than N=2"):
        build_ablation_table(records, axis=AXIS_RELEVANCE, temperature=1.0, ns=[1, 2])


def test_ablation_table_is_rendered_as_markdown() -> None:
    rows = build_ablation_table(
        _records(),
        axis=AXIS_FAITHFULNESS,
        temperature=0.7,
        ns=[1, 4],
        aggregations=["mean_prob"],
        replicates=100,
    )

    table = render_ablation_markdown(rows)

    assert table.splitlines()[0].startswith("| ось | T | N |")
    assert len(table.splitlines()) == 2 + len(rows)
    assert "mean_prob" in table
