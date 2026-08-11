"""Пофрагментный путь, доведённый до CLI и реестра.

Сам агрегат фич проверяется в ``tests/test_m3_perchunk.py``; здесь — что до
него вообще доходит вызов: ``run_m3.py --prompt-style perchunk``, метод
``m3_perchunk`` в реестре и выбор синхронного/асинхронного клиента.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import pytest

from rag_reliability.methods import registry
from rag_reliability.methods.m3.perchunk import score_per_chunk
from rag_reliability.schema import RagSample

_SPEC = importlib.util.spec_from_file_location(
    "run_m3", Path(__file__).parents[1] / "scripts" / "run_m3.py"
)
assert _SPEC is not None and _SPEC.loader is not None
run_m3 = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_m3)

PERCHUNK_KEYS = {
    "m3.max_chunk_score",
    "m3.mean_chunk_score",
    "m3.chunk_disagreement",
    "m3.n_supporting",
    "m3.argmax_chunk",
}


def make_sample(sample_id: str, n_chunks: int = 3) -> RagSample:
    context = "\n".join(f"[CHUNK {i}]\nфрагмент {i} кейса {sample_id}" for i in range(1, n_chunks + 1))
    return RagSample(
        id=sample_id,
        question=f"Клиент: вопрос {sample_id}",
        context=context,
        answer=f"ответ {sample_id}",
        faithfulness=1,
        relevance=1,
    )


def write_corpus(path: Path, samples: list[RagSample]) -> Path:
    path.write_text(
        "\n".join(json.dumps(sample.model_dump(), ensure_ascii=False) for sample in samples) + "\n",
        encoding="utf-8",
    )
    return path


def perchunk_args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "backend": "dummy",
        "model": "stub",
        "api_base": "http://localhost:8000/v1",
        "api_key_env": "OPENAI_API_KEY",
        "cache_dir": None,
        "concurrency": 1,
        "max_tokens": 400,
        "prompt_style": "perchunk",
    }
    return argparse.Namespace(**{**defaults, **overrides})


def test_run_perchunk_fills_the_five_declared_features() -> None:
    samples = [make_sample("case_a"), make_sample("case_b")]
    predictions = run_m3.run_perchunk(perchunk_args(), samples)
    assert [prediction.id for prediction in predictions] == ["case_a", "case_b"]
    for prediction in predictions:
        assert set(prediction.scores) == PERCHUNK_KEYS


def test_per_chunk_artifact_carries_no_decision() -> None:
    """Бинаризация — дело протокола; метод отдаёт фичи."""
    prediction = run_m3.run_perchunk(perchunk_args(), [make_sample("case_a")])[0]
    assert (prediction.faithfulness_pred, prediction.relevance_pred) == (0, 0)
    assert prediction.prob_method == "perchunk_logprobs"


def test_argmax_chunk_points_at_a_real_chunk() -> None:
    prediction = run_m3.run_perchunk(perchunk_args(), [make_sample("case_a", n_chunks=4)])[0]
    assert 1 <= prediction.scores["m3.argmax_chunk"] <= 4


def test_cli_writes_a_per_chunk_artifact(tmp_path: Path, monkeypatch) -> None:
    data = write_corpus(tmp_path / "corpus.jsonl", [make_sample("case_a"), make_sample("case_b")])
    output = tmp_path / "scores.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_m3.py",
            "--data", str(data),
            "--output", str(output),
            "--prompt-style", "perchunk",
            "--backend", "dummy",
        ],
    )
    run_m3.main()

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2
    assert set(rows[0]["scores"]) == PERCHUNK_KEYS


def test_perchunk_refuses_a_backend_without_verdict_logprobs() -> None:
    with pytest.raises(ValueError, match="no client for backend"):
        run_m3.build_perchunk_client(perchunk_args(backend="mlx"))


def test_concurrency_above_one_picks_the_async_client() -> None:
    """Иначе 5-15 запросов на кейс уходили бы по очереди."""
    from rag_reliability.methods.m3.judge_client import AsyncJudgeClient, JudgeClient  # noqa: PLC0415

    assert isinstance(
        run_m3.build_perchunk_client(perchunk_args(backend="openai_judge", concurrency=8)),
        AsyncJudgeClient,
    )
    assert isinstance(
        run_m3.build_perchunk_client(perchunk_args(backend="openai_judge", concurrency=1)),
        JudgeClient,
    )


# --------------------------------------------------------------------------- #
# Реестр
# --------------------------------------------------------------------------- #


def test_registry_declares_the_per_chunk_keys() -> None:
    spec = registry.get("m3_perchunk")
    assert set(spec.score_keys) == PERCHUNK_KEYS
    assert spec.default_score_expr == "m3.max_chunk_score"


def test_registry_scorer_produces_the_declared_keys(tmp_path: Path, monkeypatch) -> None:
    ctx = registry.CommandContext(
        data=tmp_path / "corpus.jsonl",
        run_dir=tmp_path,
        predictions_path=tmp_path / "scores.jsonl",
        m3_cache_dir=str(tmp_path / "cache"),
    )
    monkeypatch.setattr(registry, "build_m3_chat_client", lambda _ctx: run_m3.DummyAxisClient())
    prediction = registry.build_scorer("m3_perchunk", ctx)(make_sample("case_a"))
    assert set(prediction.scores) == PERCHUNK_KEYS


def test_command_passes_the_prompt_style_and_concurrency(tmp_path: Path) -> None:
    ctx = registry.CommandContext(
        data=tmp_path / "corpus.jsonl",
        run_dir=tmp_path,
        predictions_path=tmp_path / "scores.jsonl",
        m3_concurrency=16,
    )
    command = registry.get("m3_perchunk").build_command(ctx)
    assert "--prompt-style" in command and command[command.index("--prompt-style") + 1] == "perchunk"
    assert command[command.index("--concurrency") + 1] == "16"


def test_relevance_never_gets_chunks() -> None:
    """Контракт C3: определение relevance опирается только на вопрос и ответ."""
    with pytest.raises(ValueError, match="only axis"):
        score_per_chunk(make_sample("case_a"), run_m3.DummyAxisClient(), axis="relevance")
