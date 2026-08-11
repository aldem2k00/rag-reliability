# tests/test_score_cli.py
"""Контракт scripts/score.py: корпус-wide артефакт, продолжаемость, валидация."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import score  # noqa: E402

from rag_reliability.dataset import load_jsonl  # noqa: E402
from rag_reliability.methods import registry  # noqa: E402
from rag_reliability.schema import Prediction, RagSample  # noqa: E402

DUMMY_DATA = "data/dummy.jsonl"
DUMMY_N = 36


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _run(tmp_path: Path, *extra: str) -> Path:
    output = tmp_path / "scores.jsonl"
    exit_code = score.main(
        [
            "--method", "dummy_direct",
            "--variant", "smoke",
            "--data", DUMMY_DATA,
            "--output", str(output),
            *extra,
        ]
    )
    assert exit_code == 0
    return output


def test_scores_whole_file_and_writes_run_yaml(tmp_path: Path) -> None:
    output = _run(tmp_path)

    rows = _rows(output)
    assert len(rows) == DUMMY_N
    assert [row["id"] for row in rows] == [sample.id for sample in load_jsonl(DUMMY_DATA)]

    meta = yaml.safe_load((tmp_path / "run.yaml").read_text(encoding="utf-8"))
    assert meta["method"]["name"] == "dummy_direct"
    assert meta["variant"] == "smoke"
    assert meta["n"] == DUMMY_N
    assert meta["partial"] is False
    assert meta["seed"] == 0
    assert "hash" in meta["git"] and "dirty" in meta["git"]


def test_limit_marks_the_run_partial(tmp_path: Path) -> None:
    output = _run(tmp_path, "--limit", "5")

    assert len(_rows(output)) == 5
    meta = yaml.safe_load((tmp_path / "run.yaml").read_text(encoding="utf-8"))
    assert meta["partial"] is True
    assert meta["n"] == 5


def test_rerun_without_resume_starts_from_scratch(tmp_path: Path) -> None:
    _run(tmp_path, "--limit", "5")
    output = _run(tmp_path)

    rows = _rows(output)
    assert len(rows) == DUMMY_N
    assert len({row["id"] for row in rows}) == DUMMY_N


# --------------------------------------------------------------------------- #
# Прерывание и --resume
# --------------------------------------------------------------------------- #


def _failing_scorer(fail_at: int):
    """Скорер, падающий на fail_at-м кейсе (эмуляция обрыва сессии DataSphere)."""
    calls = {"n": 0}

    def scorer(sample: RagSample) -> Prediction:
        calls["n"] += 1
        if calls["n"] == fail_at:
            raise RuntimeError("session died")
        return Prediction(
            id=sample.id,
            faithfulness_pred=0,
            relevance_pred=0,
            scores={"ind.faith_score": 0.5, "ind.rel_score": 0.5},
        )

    return scorer


def test_partial_file_is_readable_after_interruption(tmp_path: Path) -> None:
    samples = load_jsonl(DUMMY_DATA)
    output = tmp_path / "scores.jsonl"

    with pytest.raises(RuntimeError):
        score.score_samples(
            samples, _failing_scorer(10), output, flush_every=1, progress=False
        )

    rows = _rows(output)
    assert len(rows) == 9  # девять досчитанных кейсов пережили обрыв
    assert [row["id"] for row in rows] == [sample.id for sample in samples[:9]]


def test_rows_become_visible_before_the_run_ends(tmp_path: Path) -> None:
    """Файл наполняется по ходу прогона, а не одним куском в конце.

    Иначе обрыв на 2000-м кейсе из 2245 стоит всего прогона.
    """
    samples = load_jsonl(DUMMY_DATA)
    output = tmp_path / "scores.jsonl"
    visible: list[int] = []

    def scorer(sample: RagSample) -> Prediction:
        visible.append(len(_rows(output)) if output.exists() else 0)
        return Prediction(id=sample.id, faithfulness_pred=0, relevance_pred=0)

    score.score_samples(samples, scorer, output, flush_every=5, progress=False)

    assert max(visible) >= 5  # к последним кейсам на диске уже лежат ранние
    assert visible[-1] < DUMMY_N


def test_resume_finishes_the_run_without_duplicates(tmp_path: Path) -> None:
    samples = load_jsonl(DUMMY_DATA)
    output = tmp_path / "scores.jsonl"

    with pytest.raises(RuntimeError):
        score.score_samples(
            samples, _failing_scorer(10), output, flush_every=1, progress=False
        )

    n = score.score_samples(
        samples, _failing_scorer(0), output, resume=True, flush_every=1, progress=False
    )

    rows = _rows(output)
    assert n == DUMMY_N
    assert len(rows) == DUMMY_N
    assert len({row["id"] for row in rows}) == DUMMY_N
    assert [row["id"] for row in rows] == [sample.id for sample in samples]


def test_resume_drops_a_torn_last_line(tmp_path: Path) -> None:
    """Строка без завершающего \\n — след SIGKILL посреди записи, кейс не посчитан."""
    samples = load_jsonl(DUMMY_DATA)
    output = tmp_path / "scores.jsonl"
    output.write_text(
        json.dumps({"id": samples[0].id, "scores": {}}) + "\n" + '{"id": "torn", "sco',
        encoding="utf-8",
    )

    assert score.scored_ids(output) == [samples[0].id]

    score.score_samples(
        samples, _failing_scorer(0), output, resume=True, flush_every=1, progress=False
    )
    rows = _rows(output)
    assert len(rows) == DUMMY_N
    assert "torn" not in {row["id"] for row in rows}


# --------------------------------------------------------------------------- #
# validate_scores_file
# --------------------------------------------------------------------------- #


def _write(path: Path, rows: list[dict]) -> Path:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )
    return path


def test_validate_rejects_missing_declared_score_key(tmp_path: Path) -> None:
    spec = registry.get("independent")
    path = _write(
        tmp_path / "scores.jsonl",
        [{"id": "a", "scores": {"ind.faith_score": 0.4, "ind.rel_score": 0.2}},
         {"id": "b", "scores": {"ind.faith_score": 0.4}}],
    )

    with pytest.raises(registry.ScoresValidationError, match="ind.rel_score"):
        registry.validate_scores_file(path, spec)


def test_validate_rejects_duplicate_ids(tmp_path: Path) -> None:
    spec = registry.get("dummy_direct")
    path = _write(tmp_path / "scores.jsonl", [{"id": "a", "scores": {}}, {"id": "a", "scores": {}}])

    with pytest.raises(registry.ScoresValidationError, match="Duplicate"):
        registry.validate_scores_file(path, spec)


def test_validate_rejects_non_finite_scores(tmp_path: Path) -> None:
    spec = registry.get("independent")
    path = tmp_path / "scores.jsonl"
    path.write_text(
        '{"id": "a", "scores": {"ind.faith_score": NaN, "ind.rel_score": 0.2}}\n',
        encoding="utf-8",
    )

    with pytest.raises(registry.ScoresValidationError, match="not finite"):
        registry.validate_scores_file(path, spec)


def test_validate_rejects_non_finite_undeclared_score(tmp_path: Path) -> None:
    """Незаявленный ключ с NaN — тоже брак: он попадёт в стэкер наравне с остальными.

    Контракт задаёт минимум содержимого строки, а не разрешение писать мусор мимо него.
    """
    spec = registry.get("independent")
    path = tmp_path / "scores.jsonl"
    path.write_text(
        '{"id": "a", "scores": {"ind.faith_score": 0.4, "ind.rel_score": 0.2, '
        '"ind.extra": NaN}}\n',
        encoding="utf-8",
    )

    with pytest.raises(registry.ScoresValidationError, match="ind.extra"):
        registry.validate_scores_file(path, spec)


def test_validate_rejects_non_numeric_undeclared_score(tmp_path: Path) -> None:
    spec = registry.get("independent")
    path = _write(
        tmp_path / "scores.jsonl",
        [{"id": "a", "scores": {"ind.faith_score": 0.4, "ind.rel_score": 0.2, "ind.x": "hi"}}],
    )

    with pytest.raises(registry.ScoresValidationError, match="must be a number"):
        registry.validate_scores_file(path, spec)


def test_validate_rejects_wrong_row_count(tmp_path: Path) -> None:
    spec = registry.get("dummy_direct")
    path = _write(tmp_path / "scores.jsonl", [{"id": "a", "scores": {}}])

    with pytest.raises(registry.ScoresValidationError, match="expected 2"):
        registry.validate_scores_file(path, spec, expected_n=2)


def test_validate_accepts_a_conforming_artifact(tmp_path: Path) -> None:
    spec = registry.get("independent")
    path = _write(
        tmp_path / "scores.jsonl",
        [{"id": "a", "scores": {"ind.faith_score": 0.4, "ind.rel_score": 0.2}}],
    )

    registry.validate_scores_file(path, spec, expected_n=1)


# --------------------------------------------------------------------------- #
# Методы не бинаризуют
# --------------------------------------------------------------------------- #


def _offline_scorable() -> list[str]:
    """Методы со скорером, который считается без GPU, сети и артефактов."""
    return ["dummy_direct", "dummy_marker", "independent"]


@pytest.mark.parametrize("name", _offline_scorable())
def test_scorer_does_not_binarize_except_independent(name: str, tmp_path: Path) -> None:
    ctx = registry.CommandContext(
        data=Path(DUMMY_DATA),
        run_dir=tmp_path,
        predictions_path=tmp_path / "scores.jsonl",
    )
    scorer = registry.build_scorer(name, ctx)
    predictions = [scorer(sample) for sample in load_jsonl(DUMMY_DATA)]

    if name == "independent":
        # Единственное исключение из карточки: rule-based метод бинарен по природе.
        assert any(p.faithfulness_pred or p.relevance_pred for p in predictions)
    else:
        assert all(p.faithfulness_pred == 0 and p.relevance_pred == 0 for p in predictions)


def test_m3_scorer_keeps_probabilities_out_of_binary_fields() -> None:
    """Регресс на run_m3.py:50-51: вердикт по 0.5 не должен попадать в артефакт."""
    parsed = Prediction(
        id="x", faithfulness_pred=1, relevance_pred=1, faithfulness_prob=0.83, relevance_prob=0.12
    )

    scored = registry.scores_only(parsed, registry._m3_scores(parsed))

    assert scored.faithfulness_pred == 0
    assert scored.relevance_pred == 0
    assert scored.scores == {"m3.p_faith": 0.83, "m3.p_rel": 0.12}


def test_m3_scores_fall_back_to_the_parsed_verdict_then_to_half() -> None:
    parsed = Prediction(id="x", faithfulness_pred=1, relevance_pred=0)
    assert registry._m3_scores(parsed) == {"m3.p_faith": 1.0, "m3.p_rel": 0.0}

    unparsed = Prediction(id="x", faithfulness_pred=0, relevance_pred=0, invalid_output=True)
    assert registry._m3_scores(unparsed) == {"m3.p_faith": 0.5, "m3.p_rel": 0.5}


def test_independent_scorer_fills_declared_keys(tmp_path: Path) -> None:
    spec = registry.get("independent")
    output = tmp_path / "scores.jsonl"
    ctx = registry.CommandContext(
        data=Path(DUMMY_DATA), run_dir=tmp_path, predictions_path=output
    )

    score.score_samples(
        load_jsonl(DUMMY_DATA),
        registry.build_scorer("independent", ctx),
        output,
        progress=False,
    )

    registry.validate_scores_file(output, spec, expected_n=DUMMY_N)


def test_score_cli_refuses_a_method_without_a_corpus_wide_scorer(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="C2"):
        score.main(
            [
                "--method", "encoder",
                "--variant", "oof",
                "--data", DUMMY_DATA,
                "--output", str(tmp_path / "scores.jsonl"),
            ]
        )


# --------------------------------------------------------------------------- #
# Параллельный счёт: --workers
# --------------------------------------------------------------------------- #


def test_workers_preserve_corpus_order(tmp_path: Path) -> None:
    """Порядок обязателен: --resume дочитывает файл сверху.

    Если строки лягут «как посчиталось», обрыв оставит дыру в середине, а
    перезапуск её не заметит.
    """
    output = _run(tmp_path, "--workers", "8")
    assert [row["id"] for row in _rows(output)] == [
        sample.id for sample in load_jsonl(DUMMY_DATA)
    ]


def test_workers_give_the_same_artifact_as_a_sequential_run(tmp_path: Path) -> None:
    sequential = _rows(_run(tmp_path / "seq"))
    parallel = _rows(_run(tmp_path / "par", "--workers", "4"))
    assert sequential == parallel


def test_workers_actually_run_concurrently(tmp_path: Path) -> None:
    """Иначе флаг был бы косметикой: последовательный цикл под другим именем."""
    import threading

    output = tmp_path / "scores.jsonl"
    samples = load_jsonl(DUMMY_DATA)
    barrier = threading.Barrier(4, timeout=10)

    def slow_scorer(sample: RagSample) -> Prediction:
        # Проходит только если четыре потока оказались внутри одновременно.
        barrier.wait()
        return Prediction(id=sample.id, faithfulness_pred=0, relevance_pred=0)

    n = score.score_samples(samples, slow_scorer, output, progress=False, workers=4)
    assert n == len(samples)


def test_resume_after_a_parallel_run_scores_the_rest(tmp_path: Path) -> None:
    output = tmp_path / "scores.jsonl"
    samples = load_jsonl(DUMMY_DATA)
    ctx = registry.CommandContext(
        data=Path(DUMMY_DATA), run_dir=tmp_path, predictions_path=output
    )
    scorer = registry.build_scorer("dummy_direct", ctx)

    score.score_samples(samples[:10], scorer, output, progress=False, workers=4)
    total = score.score_samples(samples, scorer, output, progress=False, resume=True, workers=4)

    assert total == len(samples)
    assert [row["id"] for row in _rows(output)] == [sample.id for sample in samples]


def test_workers_below_one_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workers must be >= 1"):
        score.score_samples(
            load_jsonl(DUMMY_DATA),
            lambda sample: Prediction(id=sample.id, faithfulness_pred=0, relevance_pred=0),
            tmp_path / "scores.jsonl",
            progress=False,
            workers=0,
        )


def test_m3_concurrency_reaches_the_command_context() -> None:
    """Флаг обязан доезжать до реестра: иначе он молча ничего не делает."""
    args = score.parse_args(
        [
            "--method", "m3_perchunk",
            "--variant", "smoke",
            "--data", DUMMY_DATA,
            "--output", "scores.jsonl",
            "--m3-concurrency", "16",
        ]
    )
    assert score.build_context(args).m3_concurrency == 16
