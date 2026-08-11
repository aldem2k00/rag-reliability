"""OOF-раскладка энкодера и контроль схлопывания — на подставном тренере.

Ни torch, ни transformers, ни единого скачанного веса: обучение фолда приходит
параметром, а тест подставляет функцию, которая записывает, что ей дали.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from rag_reliability.methods import registry
from rag_reliability.methods.encoder.predict import (
    LOGIT_KEY,
    PROB_KEY,
    checkpoint_meta,
    logits_to_predictions,
    sigmoid,
    write_scores,
)
from rag_reliability.methods.encoder.train import (
    FoldOutcome,
    FoldRequest,
    OofResult,
    TrainConfig,
    compute_pos_weight,
    decisions_from_logits,
    is_collapsed,
    train_oof,
    train_oof_detailed,
)
from rag_reliability.methods.surface.oof import Folds
from rag_reliability.schema import RagSample


def make_sample(sample_id: str, faithfulness: int = 1, relevance: int = 1) -> RagSample:
    return RagSample(
        id=sample_id,
        question=f"Клиент: вопрос {sample_id}",
        context=f"[CHUNK 1]\nконтекст {sample_id}",
        answer=f"ответ {sample_id}",
        faithfulness=faithfulness,
        relevance=relevance,
        marker="none" if faithfulness and relevance else "unknown",
    )


def make_corpus(n: int = 20) -> list[RagSample]:
    # Каждый пятый ненадёжен: без обоих классов диагностика схлопывания слепа.
    return [make_sample(f"case_{index:03d}", relevance=int(index % 5 != 0)) for index in range(n)]


def make_folds(samples: list[RagSample], *, n_folds: int = 5, n_repeats: int = 2) -> Folds:
    assignment = {
        sample.id: [(index + repeat) % n_folds for repeat in range(n_repeats)]
        for index, sample in enumerate(samples)
    }
    return Folds(
        assignment=assignment,
        n_folds=n_folds,
        n_repeats=n_repeats,
        corpus_n=len(samples),
        sha256="0" * 64,
    )


class RecordingTrainer:
    """Подставной тренер: помнит, что видел, и честно зовёт хук после эпохи."""

    def __init__(self, *, logit: float | None = None, epochs_to_report: int | None = None) -> None:
        self.seen: list[tuple[int, tuple[str, ...], tuple[str, ...]]] = []
        self.extra_seen: list[tuple[str, ...]] = []
        self.logit = logit
        self.epochs_to_report = epochs_to_report

    def __call__(self, request: FoldRequest) -> FoldOutcome:
        train_ids = tuple(sample.id for sample in request.train_samples)
        test_ids = tuple(sample.id for sample in request.test_samples)
        self.seen.append((request.fold, train_ids, test_ids))
        self.extra_seen.append(tuple(sample.id for sample in request.extra_samples))
        # Логит по умолчанию зависит от кейса, чтобы прогон не выглядел схлопнувшимся.
        logits = [
            self.logit if self.logit is not None else (1.0 if index % 2 else -1.0)
            for index in range(len(test_ids))
        ]
        n_epochs = (
            self.epochs_to_report
            if self.epochs_to_report is not None
            else request.config.n_epochs
        )
        for epoch in range(1, n_epochs + 1):
            request.on_epoch_end(epoch, logits)
        return FoldOutcome(
            logits=tuple(logits),
            checkpoint=f"ckpt/fold{request.fold}.pt",
            # Номер фолда: так видно, что усреднение идёт по всем моделям.
            extra_logits=tuple(float(request.fold) for _ in request.extra_samples),
        )


# --------------------------------------------------------------------------- #
# Конфигурация
# --------------------------------------------------------------------------- #


def test_config_rejects_unknown_class_weighting() -> None:
    with pytest.raises(ValueError, match="pos_weight_mode"):
        TrainConfig(pos_weight_mode="inverse")


def test_config_rejects_warmup_ratio_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="warmup_ratio"):
        TrainConfig(warmup_ratio=1.0)


def test_config_rounds_fractional_epochs_up_for_diagnostics() -> None:
    assert TrainConfig(epochs=2.5).n_epochs == 3
    assert TrainConfig(epochs=3.0).n_epochs == 3


def test_pos_weight_balances_the_positive_majority() -> None:
    assert compute_pos_weight([1, 1, 1, 0], mode="balanced") == 1 / 3
    assert compute_pos_weight([1, 1, 1, 0], mode="none") == 1.0


# --------------------------------------------------------------------------- #
# Изоляция фолдов
# --------------------------------------------------------------------------- #


def test_train_oof_never_shows_a_fold_its_own_cases() -> None:
    samples = make_corpus()
    folds = make_folds(samples)
    trainer = RecordingTrainer()

    train_oof(samples, folds, TrainConfig(), repeat=0, train_fold=trainer)

    assert len(trainer.seen) == folds.n_folds
    for fold, train_ids, test_ids in trainer.seen:
        assert set(train_ids) & set(test_ids) == set()
        expected = {
            sample.id for sample in samples if folds.assignment[sample.id][0] == fold
        }
        assert set(test_ids) == expected
        assert set(train_ids) | set(test_ids) == {sample.id for sample in samples}


def test_train_oof_uses_the_requested_repeat() -> None:
    samples = make_corpus()
    folds = make_folds(samples)
    trainer = RecordingTrainer()

    train_oof(samples, folds, TrainConfig(), repeat=1, train_fold=trainer)

    for fold, _train_ids, test_ids in trainer.seen:
        expected = {sample.id for sample in samples if folds.assignment[sample.id][1] == fold}
        assert set(test_ids) == expected


def test_train_oof_scores_every_case_exactly_once() -> None:
    samples = make_corpus()

    logits = train_oof(samples, make_folds(samples), TrainConfig(), train_fold=RecordingTrainer())

    assert set(logits) == {sample.id for sample in samples}


def test_train_oof_rejects_a_repeat_folds_json_does_not_have() -> None:
    samples = make_corpus()
    folds = make_folds(samples, n_repeats=1)

    with pytest.raises(ValueError, match="repeat must be in"):
        train_oof(samples, folds, TrainConfig(), repeat=3, train_fold=RecordingTrainer())


def test_cases_without_a_fold_are_scored_by_the_mean_of_the_fold_models() -> None:
    """Кейсы вне folds.json — одна oversized-группа; выбрасывать их нельзя.

    Группа целиком отсутствует в train-части любого фолда, поэтому изоляция
    здесь даже строже обычного OOF: модель не видела ни кейс, ни его группу.
    """
    samples = make_corpus()
    folds = make_folds(samples)
    orphan = make_sample("case_999")
    trainer = RecordingTrainer()

    result = train_oof_detailed(
        [*samples, orphan], folds, TrainConfig(), train_fold=trainer
    )

    assert set(result.logits) == {sample.id for sample in samples} | {orphan.id}
    assert result.ensemble_ids == (orphan.id,)
    assert set(result.oof_ids) == {sample.id for sample in samples}
    # RecordingTrainer отдаёт номер фолда: среднее по 0..4 равно 2.0.
    assert result.logits[orphan.id] == pytest.approx(2.0)
    assert all(seen == (orphan.id,) for seen in trainer.extra_seen)


def test_the_outside_group_never_enters_a_training_part() -> None:
    samples = make_corpus()
    orphan = make_sample("case_999")
    trainer = RecordingTrainer()

    train_oof([*samples, orphan], make_folds(samples), TrainConfig(), train_fold=trainer)

    for _fold, train_ids, test_ids in trainer.seen:
        assert orphan.id not in train_ids
        assert orphan.id not in test_ids


def test_a_trainer_that_skips_the_outside_cases_fails_the_run() -> None:
    """Молчаливо неполный артефакт — то, из-за чего критерий приёмки не прошёл."""
    samples = make_corpus()
    orphan = make_sample("case_999")

    def forgetful(request: FoldRequest) -> FoldOutcome:
        logits = [1.0 if index % 2 else -1.0 for index in range(len(request.test_samples))]
        for epoch in range(1, request.config.n_epochs + 1):
            request.on_epoch_end(epoch, logits)
        return FoldOutcome(logits=tuple(logits))

    with pytest.raises(ValueError, match="outside the folds"):
        train_oof([*samples, orphan], make_folds(samples), TrainConfig(), train_fold=forgetful)


def test_collapse_is_judged_on_the_out_of_fold_part_only() -> None:
    """Ансамблевые строки не должны разбавлять вердикт о схлопывании."""
    samples = make_corpus()
    orphans = [make_sample(f"case_{900 + index}") for index in range(40)]
    folds = make_folds(samples)

    def collapsed_oof(request: FoldRequest) -> FoldOutcome:
        logits = [5.0] * len(request.test_samples)
        for epoch in range(1, request.config.n_epochs + 1):
            request.on_epoch_end(epoch, logits)
        # Вне фолдов — разнобой, которого хватило бы, чтобы пул выглядел живым.
        extra = [5.0 if index % 2 else -5.0 for index in range(len(request.extra_samples))]
        return FoldOutcome(logits=tuple(logits), extra_logits=tuple(extra))

    result = train_oof_detailed(
        [*samples, *orphans], folds, TrainConfig(), train_fold=collapsed_oof
    )

    assert result.collapsed is True
    assert result.collapse_reason == "pooled"
    assert result.diagnostics()["n_scored"] == len(samples) + len(orphans)
    assert result.diagnostics()["n_oof"] == len(samples)
    assert result.diagnostics()["n_ensemble"] == len(orphans)


def test_train_oof_rejects_a_corpus_with_no_fold_assignment_at_all() -> None:
    orphans = [make_sample(f"case_{900 + index}") for index in range(5)]
    folds = make_folds(make_corpus())

    with pytest.raises(ValueError, match="None of the"):
        train_oof(orphans, folds, TrainConfig(), train_fold=RecordingTrainer())


def test_train_oof_rejects_a_trainer_that_returns_the_wrong_number_of_logits() -> None:
    samples = make_corpus()

    def short_trainer(request: FoldRequest) -> FoldOutcome:
        for epoch in range(1, request.config.n_epochs + 1):
            request.on_epoch_end(epoch, [0.5] * len(request.test_samples))
        return FoldOutcome(logits=(0.5,))

    with pytest.raises(ValueError, match="logit"):
        train_oof(samples, make_folds(samples), TrainConfig(), train_fold=short_trainer)


# --------------------------------------------------------------------------- #
# Контроль схлопывания
# --------------------------------------------------------------------------- #


def test_diagnostics_run_after_every_epoch() -> None:
    samples = make_corpus()
    folds = make_folds(samples)
    config = TrainConfig(epochs=3)

    result = train_oof_detailed(samples, folds, config, train_fold=RecordingTrainer())

    assert len(result.epochs) == folds.n_folds * config.n_epochs
    assert sorted({log.epoch for log in result.epochs}) == [1, 2, 3]
    for log in result.epochs:
        assert 0.0 < log.const_share <= 1.0


def test_a_trainer_that_skips_the_epoch_hook_fails_the_run() -> None:
    """Диагностика обязательна: молча пропущенная эпоха — это снова прогон 1024."""
    samples = make_corpus()

    with pytest.raises(ValueError, match="degenerate_rate is mandatory"):
        train_oof(
            samples,
            make_folds(samples),
            TrainConfig(epochs=3),
            train_fold=RecordingTrainer(epochs_to_report=1),
        )


def test_epoch_diagnostics_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    samples = make_corpus()

    with caplog.at_level(logging.INFO, logger="rag_reliability.methods.encoder.train"):
        train_oof(samples, make_folds(samples), TrainConfig(epochs=1), train_fold=RecordingTrainer())

    assert any("const_share" in record.getMessage() for record in caplog.records)


def test_a_run_that_predicts_one_class_everywhere_is_marked_collapsed() -> None:
    samples = make_corpus()

    result = train_oof_detailed(
        samples, make_folds(samples), TrainConfig(), train_fold=RecordingTrainer(logit=5.0)
    )

    assert result.collapsed is True
    assert result.diagnostics()["collapsed"] is True
    assert result.diagnostics()["const_share"] == 1.0
    assert all(log.is_degenerate for log in result.epochs)


def test_a_run_where_every_fold_predicts_one_class_is_collapsed() -> None:
    """Пул может выглядеть здоровым, когда разные фолды схлопнулись в разные классы.

    Воспроизведено смоуком на крошечной модели: три фолда с const_share = 1.0
    каждый, а склеенный OOF — 0.67. Обучения не было ни в одном фолде.
    """
    samples = make_corpus()

    def per_fold_constant(request: FoldRequest) -> FoldOutcome:
        # Чётные фолды — «всё надёжно», нечётные — «всё ненадёжно».
        logit = 5.0 if request.fold % 2 == 0 else -5.0
        logits = [logit] * len(request.test_samples)
        for epoch in range(1, request.config.n_epochs + 1):
            request.on_epoch_end(epoch, logits)
        return FoldOutcome(logits=tuple(logits))

    result = train_oof_detailed(
        samples, make_folds(samples), TrainConfig(), train_fold=per_fold_constant
    )

    assert result.diagnostics()["const_share"] < 0.98  # пул выглядит приличным
    assert result.collapsed is True
    assert result.collapse_reason == "per_fold"
    assert result.collapsed_folds == [0, 1, 2, 3, 4]


def test_a_pooled_collapse_is_named_as_such() -> None:
    samples = make_corpus()

    result = train_oof_detailed(
        samples, make_folds(samples), TrainConfig(), train_fold=RecordingTrainer(logit=5.0)
    )

    assert result.collapse_reason == "pooled"


def test_a_single_collapsed_fold_does_not_condemn_the_whole_run() -> None:
    """Один плохой фолд — повод посмотреть в лог, а не выбросить конфигурацию."""
    samples = make_corpus()

    def one_bad_fold(request: FoldRequest) -> FoldOutcome:
        if request.fold == 0:
            logits = [5.0] * len(request.test_samples)
        else:
            logits = [1.0 if index % 2 else -1.0 for index in range(len(request.test_samples))]
        for epoch in range(1, request.config.n_epochs + 1):
            request.on_epoch_end(epoch, logits)
        return FoldOutcome(logits=tuple(logits))

    result = train_oof_detailed(
        samples, make_folds(samples), TrainConfig(), train_fold=one_bad_fold
    )

    assert result.collapsed is False
    assert result.collapsed_folds == [0]


def test_collapsed_folds_are_judged_by_the_last_epoch() -> None:
    """Схлопывание на первой эпохе, из которого модель вышла, — не приговор."""
    samples = make_corpus()

    def recovers(request: FoldRequest) -> FoldOutcome:
        constant = [5.0] * len(request.test_samples)
        mixed = [1.0 if index % 2 else -1.0 for index in range(len(request.test_samples))]
        request.on_epoch_end(1, constant)
        request.on_epoch_end(2, mixed)
        return FoldOutcome(logits=tuple(mixed))

    result = train_oof_detailed(
        samples, make_folds(samples), TrainConfig(epochs=2), train_fold=recovers
    )

    assert result.collapsed_folds == []
    assert result.collapsed is False


def test_a_run_with_a_mixed_output_is_not_collapsed() -> None:
    samples = make_corpus()

    result = train_oof_detailed(
        samples, make_folds(samples), TrainConfig(), train_fold=RecordingTrainer()
    )

    assert result.collapsed is False
    assert result.diagnostics()["const_share"] < 0.98


def test_collapse_is_detected_at_the_threshold_from_metrics() -> None:
    """Порог 0.98 не дублируется в пакете — он приходит из metrics.degenerate_rate."""
    below = {f"case_{index}": (5.0 if index >= 2 else -5.0) for index in range(100)}
    above = {f"case_{index}": (5.0 if index >= 1 else -5.0) for index in range(100)}

    assert is_collapsed(below) is False  # const_share = 0.98, порог строгий
    assert is_collapsed(above) is True  # const_share = 0.99


def test_decisions_from_logits_binarize_at_zero() -> None:
    decisions = decisions_from_logits({"a": 0.1, "b": -0.1, "c": 0.0})

    assert [prediction.reliable_pred for prediction in decisions] == [1, 0, 1]


def test_decisions_from_an_empty_run_are_an_error_not_a_silent_pass() -> None:
    with pytest.raises(ValueError, match="empty"):
        decisions_from_logits({})


def test_diagnostics_record_the_single_repeat_explicitly() -> None:
    """n_repeats: 1 обязано быть видно при сравнении с методами, у которых 5."""
    samples = make_corpus()

    result = train_oof_detailed(
        samples, make_folds(samples), TrainConfig(), repeat=1, train_fold=RecordingTrainer()
    )

    diagnostics = result.diagnostics()
    assert diagnostics["n_repeats"] == 1
    assert diagnostics["repeat"] == 1
    assert diagnostics["n_scored"] == len(samples)


def test_checkpoints_are_recorded_per_fold() -> None:
    samples = make_corpus()
    folds = make_folds(samples)

    result = train_oof_detailed(samples, folds, TrainConfig(), train_fold=RecordingTrainer())

    assert sorted(result.checkpoints) == list(range(folds.n_folds))


def test_oof_result_of_an_empty_run_cannot_claim_a_verdict() -> None:
    with pytest.raises(ValueError, match="empty"):
        OofResult(logits={}).collapsed  # noqa: B018


# --------------------------------------------------------------------------- #
# Артефакт scores.jsonl
# --------------------------------------------------------------------------- #


def test_artifact_carries_the_raw_logit_under_the_contract_key() -> None:
    predictions = logits_to_predictions({"a": 1.5, "b": -2.0})

    assert [prediction.scores[LOGIT_KEY] for prediction in predictions] == [1.5, -2.0]
    assert set(predictions[0].scores) == {LOGIT_KEY, PROB_KEY}


def test_artifact_leaves_binarization_to_the_protocol() -> None:
    """Порог подбирается внутри train-части фолда, а не вшивается в артефакт."""
    predictions = logits_to_predictions({"a": 9.0, "b": -9.0})

    assert all(prediction.faithfulness_pred == 0 for prediction in predictions)
    assert all(prediction.relevance_pred == 0 for prediction in predictions)


def test_probability_key_is_a_monotone_image_of_the_logit_inside_the_unit_interval() -> None:
    """evaluate_cv отвергает скор вне [0, 1], а --score-expr не умеет сигмоиду."""
    logits = [-800.0, -3.0, -0.5, 0.0, 0.5, 3.0, 800.0]

    probabilities = [sigmoid(logit) for logit in logits]

    assert probabilities == sorted(probabilities)
    assert all(0.0 <= probability <= 1.0 for probability in probabilities)
    assert sigmoid(0.0) == 0.5


def test_artifact_rejects_a_non_finite_logit() -> None:
    with pytest.raises(ValueError, match="not finite"):
        logits_to_predictions({"a": float("nan")})


def test_artifact_of_an_empty_run_is_an_error() -> None:
    with pytest.raises(ValueError, match="empty"):
        logits_to_predictions({})


def test_written_artifact_satisfies_the_registry_contract(tmp_path: Path) -> None:
    samples = make_corpus()
    logits = train_oof(samples, make_folds(samples), TrainConfig(), train_fold=RecordingTrainer())
    path = tmp_path / "scores.jsonl"

    n_written = write_scores(logits_to_predictions(logits), path)

    registry.validate_scores_file(path, registry.get("encoder"), expected_n=len(samples))
    assert n_written == len(samples)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert all(row["scores"][LOGIT_KEY] == logits[row["id"]] for row in rows)


def test_checkpoint_meta_reports_a_missing_file_instead_of_a_fake_hash(tmp_path: Path) -> None:
    present = tmp_path / "fold0.pt"
    present.write_bytes(b"weights")

    meta = checkpoint_meta({0: str(present), 1: str(tmp_path / "fold1.pt")})

    assert meta[0]["exists"] is True
    assert len(meta[0]["sha256"]) == 64
    assert meta[1]["exists"] is False
    assert meta[1]["sha256"] is None
