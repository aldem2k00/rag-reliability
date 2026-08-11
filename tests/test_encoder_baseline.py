"""CLI энкодера: скрипт — обёртка, собственного сплита в нём нет.

Прогон идёт на подставном тренере: ни torch, ни transformers, ни весов.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from rag_reliability.dataset import save_jsonl
from rag_reliability.methods.encoder.predict import LOGIT_KEY, PROB_KEY
from rag_reliability.methods.encoder.train import FoldOutcome, FoldRequest
from rag_reliability.schema import RagSample

_SPEC = importlib.util.spec_from_file_location(
    "train_encoder_baseline",
    Path(__file__).parents[1] / "scripts" / "train_encoder_baseline.py",
)
assert _SPEC is not None
encoder_baseline = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(encoder_baseline)

REPO_ROOT = Path(__file__).parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "rag_reliability" / "methods" / "encoder"


def make_sample(index: int) -> RagSample:
    relevance = int(index % 5 != 0)
    return RagSample(
        id=f"case_{index:03d}",
        question=f"Клиент: вопрос {index}",
        context=f"[CHUNK 1]\nконтекст {index}",
        answer=f"ответ {index}",
        faithfulness=1,
        relevance=relevance,
        marker="none" if relevance else "unknown",
    )


def alternating_trainer(request: FoldRequest) -> FoldOutcome:
    logits = [1.0 if index % 2 else -1.0 for index in range(len(request.test_samples))]
    for epoch in range(1, request.config.n_epochs + 1):
        request.on_epoch_end(epoch, logits)
    return FoldOutcome(
        logits=tuple(logits),
        extra_logits=tuple(float(request.fold) for _ in request.extra_samples),
    )


def constant_trainer(request: FoldRequest) -> FoldOutcome:
    logits = [3.0] * len(request.test_samples)
    for epoch in range(1, request.config.n_epochs + 1):
        request.on_epoch_end(epoch, logits)
    return FoldOutcome(
        logits=tuple(logits), extra_logits=tuple(3.0 for _ in request.extra_samples)
    )


@pytest.fixture
def corpus(tmp_path: Path) -> dict[str, Any]:
    """Корпус + folds.json, где часть кейсов вне фолдов (как oversized-группа)."""
    samples = [make_sample(index) for index in range(20)]
    outside = [make_sample(index) for index in range(100, 104)]
    data_path = tmp_path / "corpus.jsonl"
    save_jsonl([*samples, *outside], data_path)

    folds_path = tmp_path / "folds.json"
    folds_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "corpus": {"path": str(data_path), "sha256": "0" * 64, "n": len(samples)},
                "config": {"n_folds": 5, "n_repeats": 2, "seed": 0},
                "assignment": {
                    sample.id: [index % 5, (index + 1) % 5]
                    for index, sample in enumerate(samples)
                },
            }
        ),
        encoding="utf-8",
    )
    return {
        "data": str(data_path),
        "folds": str(folds_path),
        "n_in_folds": len(samples),
        "n_corpus": len(samples) + len(outside),
        "ids_in_folds": [sample.id for sample in samples],
        "ids_outside": [sample.id for sample in outside],
        "all_ids": [sample.id for sample in (*samples, *outside)],
    }


def run_cli(corpus: dict[str, Any], tmp_path: Path, *extra: str, trainer: Any) -> Path:
    scores_path = tmp_path / "run" / "scores.jsonl"
    encoder_baseline.main(
        [
            "--data", corpus["data"],
            "--folds", corpus["folds"],
            "--predictions-output", str(scores_path),
            "--epochs", "2",
            *extra,
        ],
        train_fold=trainer,
    )
    return scores_path


# --------------------------------------------------------------------------- #
# Скрипт больше не сплитит
# --------------------------------------------------------------------------- #


def test_the_script_no_longer_defines_its_own_split() -> None:
    source = (REPO_ROOT / "scripts" / "train_encoder_baseline.py").read_text(encoding="utf-8")

    assert "split_samples" not in source
    assert "train_test_split" not in source


def test_the_package_never_calls_split_samples() -> None:
    """Четвёртого протокола разбиения в кодовой базе быть не должно (AGENTS §П4)."""
    offenders = [
        path.name
        for path in PACKAGE_ROOT.glob("*.py")
        if "split_samples" in path.read_text(encoding="utf-8")
    ]

    assert offenders == []


def test_the_script_reads_folds_from_the_canonical_corpus_split() -> None:
    """Числа сравнимы только внутри одного folds.json — того же, что у surface."""
    args = encoder_baseline.parse_args([])

    assert args.data == "data/alfa.jsonl"
    assert args.folds == "data/splits/folds_alfa.json"


# --------------------------------------------------------------------------- #
# Артефакты прогона
# --------------------------------------------------------------------------- #


def test_cli_writes_one_scores_row_per_corpus_case(
    corpus: dict[str, Any], tmp_path: Path
) -> None:
    """Артефакт покрывает корпус целиком, а не только кейсы внутри фолдов.

    Кейсы вне фолдов — одна oversized-группа, отсутствующая в train-части
    любого фолда; выбросить их значит отдать стэкеру треть корпуса без скора.
    """
    scores_path = run_cli(corpus, tmp_path, trainer=alternating_trainer)

    rows = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == corpus["n_corpus"]
    assert {row["id"] for row in rows} == set(corpus["all_ids"])
    assert all(set(row["scores"]) == {LOGIT_KEY, PROB_KEY} for row in rows)
    assert all(row["faithfulness_pred"] == 0 and row["relevance_pred"] == 0 for row in rows)


def test_cases_outside_the_folds_are_marked_as_ensemble_scored(
    corpus: dict[str, Any], tmp_path: Path
) -> None:
    """Строку с ансамблевым скором нельзя спутать с честной out-of-fold."""
    scores_path = run_cli(corpus, tmp_path, trainer=alternating_trainer)

    rows = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines()]
    by_method: dict[str, set[str]] = {}
    for row in rows:
        by_method.setdefault(row["prob_method"], set()).add(row["id"])

    assert by_method["encoder_oof"] == set(corpus["ids_in_folds"])
    assert by_method["encoder_fold_ensemble"] == set(corpus["ids_outside"])


def test_ensemble_logit_is_the_mean_over_fold_models(
    corpus: dict[str, Any], tmp_path: Path
) -> None:
    """alternating_trainer отдаёт номер фолда: среднее по 0..4 — это 2.0."""
    scores_path = run_cli(corpus, tmp_path, trainer=alternating_trainer)

    rows = [json.loads(line) for line in scores_path.read_text(encoding="utf-8").splitlines()]
    outside = [row for row in rows if row["id"] in set(corpus["ids_outside"])]

    assert outside
    assert all(row["scores"][LOGIT_KEY] == pytest.approx(2.0) for row in outside)


def test_a_trainer_that_ignores_the_outside_cases_fails_the_run(
    corpus: dict[str, Any], tmp_path: Path
) -> None:
    """Молча неполный артефакт — это ровно то, из-за чего C2 не прошла приёмку."""

    def forgetful(request: FoldRequest) -> FoldOutcome:
        logits = [1.0 if index % 2 else -1.0 for index in range(len(request.test_samples))]
        for epoch in range(1, request.config.n_epochs + 1):
            request.on_epoch_end(epoch, logits)
        return FoldOutcome(logits=tuple(logits))

    with pytest.raises(ValueError, match="outside the folds"):
        run_cli(corpus, tmp_path, trainer=forgetful)


def test_run_yaml_records_the_single_repeat_and_the_collapse_verdict(
    corpus: dict[str, Any], tmp_path: Path
) -> None:
    scores_path = run_cli(corpus, tmp_path, trainer=alternating_trainer)

    payload = yaml.safe_load((scores_path.parent / "run.yaml").read_text(encoding="utf-8"))
    encoder = payload["encoder"]
    assert encoder["n_repeats"] == 1
    assert encoder["collapsed"] is False
    assert encoder["max_length"] == 512
    assert encoder["warmup_ratio"] == 0.06
    assert encoder["const_share"] < 0.98


def test_run_yaml_marks_a_collapsed_configuration(
    corpus: dict[str, Any], tmp_path: Path
) -> None:
    """Схлопнувшийся прогон обязан быть отличим от нормального без чтения метрик."""
    scores_path = run_cli(corpus, tmp_path, trainer=constant_trainer)

    payload = yaml.safe_load((scores_path.parent / "run.yaml").read_text(encoding="utf-8"))
    assert payload["encoder"]["collapsed"] is True
    assert payload["encoder"]["const_share"] == 1.0


def test_run_yaml_states_coverage_and_splits_it_by_score_source(
    corpus: dict[str, Any], tmp_path: Path
) -> None:
    scores_path = run_cli(corpus, tmp_path, trainer=alternating_trainer)

    payload = yaml.safe_load((scores_path.parent / "run.yaml").read_text(encoding="utf-8"))
    coverage = payload["coverage"]
    assert coverage["corpus_n"] == corpus["n_corpus"]
    assert coverage["scored_n"] == corpus["n_corpus"]
    assert coverage["excluded_n"] == 0
    assert coverage["oof_n"] == corpus["n_in_folds"]
    assert coverage["fold_ensemble_n"] == corpus["n_corpus"] - corpus["n_in_folds"]
    assert len(coverage["corpus_sha256"]) == 64
    assert payload["partial"] is False


def test_epoch_diagnostics_land_in_the_run_artifacts(
    corpus: dict[str, Any], tmp_path: Path
) -> None:
    scores_path = run_cli(corpus, tmp_path, trainer=alternating_trainer)

    diagnostics = json.loads(
        (scores_path.parent / "encoder_diagnostics.json").read_text(encoding="utf-8")
    )
    assert len(diagnostics["epochs"]) == 5 * 2  # 5 фолдов × 2 эпохи
    assert {log["epoch"] for log in diagnostics["epochs"]} == {1, 2}
    assert all("const_share" in log for log in diagnostics["epochs"])


def test_cli_honours_the_hyperparameters_it_was_given(
    corpus: dict[str, Any], tmp_path: Path
) -> None:
    seen: list[Any] = []

    def recording_trainer(request: FoldRequest) -> FoldOutcome:
        seen.append(request.config)
        return alternating_trainer(request)

    run_cli(
        corpus,
        tmp_path,
        "--max-length", "8192",
        "--learning-rate", "1e-5",
        "--grad-accum", "8",
        "--batch-size", "1",
        "--pos-weight-mode", "balanced",
        trainer=recording_trainer,
    )

    config = seen[0]
    assert config.max_length == 8192
    assert config.learning_rate == 1e-5
    assert config.grad_accum == 8
    assert config.pos_weight_mode == "balanced"


def test_limit_runs_a_smoke_over_the_first_cases(
    corpus: dict[str, Any], tmp_path: Path
) -> None:
    scores_path = run_cli(corpus, tmp_path, "--limit", "10", trainer=alternating_trainer)

    rows = scores_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 10


def test_default_artifact_path_follows_the_method_layout() -> None:
    args = encoder_baseline.parse_args(["--variant", "len8192_lr2e-5"])
    scores_path, run_yaml, diagnostics = encoder_baseline.resolve_paths(args)

    assert scores_path == Path("predictions/alfa/encoder/len8192_lr2e-5/scores.jsonl")
    assert run_yaml == scores_path.parent / "run.yaml"
    assert diagnostics == scores_path.parent / "encoder_diagnostics.json"
