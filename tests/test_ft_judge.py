"""FT судьи: формат, баланс классов, раскладка по фолдам и контроль схлопывания.

Ни torch, ни transformers, ни единого скачанного веса: обучение фолда приходит
параметром, а тест подставляет функцию, которая записывает, что ей дали. Ровно
те части, которые ронял прошлый прогон (симметрия формата и дисбаланс 72/28),
проверяются без GPU.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from rag_reliability.methods import registry
from rag_reliability.methods.ft_judge.data import (
    JudgeExample,
    build_examples,
    check_format_symmetry,
    class_balance,
    completion_text,
    compute_pos_weight,
    gold_marker,
    oversample_negatives,
)
from rag_reliability.methods.ft_judge.predict import (
    FAITH_KEY,
    REL_KEY,
    AxisProbs,
    decisions_from_probs,
    probs_to_predictions,
)
from rag_reliability.methods.ft_judge.train import (
    FoldOutcome,
    FoldRequest,
    FoldResult,
    FtConfig,
    fold_partition,
    train_one_fold,
)
from rag_reliability.methods.surface.oof import Folds
from rag_reliability.schema import RagSample

_SPEC = importlib.util.spec_from_file_location(
    "train_ft_judge", Path(__file__).parents[1] / "scripts" / "train_ft_judge.py"
)
assert _SPEC is not None and _SPEC.loader is not None
train_ft_judge = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(train_ft_judge)


def make_sample(sample_id: str, faithfulness: int = 1, relevance: int = 1) -> RagSample:
    return RagSample(
        id=sample_id,
        question=f"Клиент: вопрос {sample_id}",
        context=f"[CHUNK 1]\nконтекст {sample_id}",
        answer=f"ответ {sample_id}",
        faithfulness=faithfulness,
        relevance=relevance,
        marker="none" if faithfulness and relevance else "unsupported_claim",
    )


def make_corpus(n: int = 20) -> list[RagSample]:
    """Обе оси с дефектами, и период дефекта не совпадает с числом фолдов.

    Период 5 при 5 фолдах отправил бы все негативы ровно в один фолд: у
    остальных train-часть осталась бы одноклассовой, и взвешивание с
    oversampling проверялись бы на вырожденных данных.
    """
    return [
        make_sample(
            f"case_{index:03d}",
            faithfulness=int(index % 4 != 0),
            relevance=int(index % 3 != 0),
        )
        for index in range(n)
    ]


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
    """Подставной тренер: отдаёт заданные вероятности и логирует каждую эпоху."""

    def __init__(self, probs: list[tuple[float, float]] | None = None, *, epochs: int = 3) -> None:
        self.probs = probs
        self.epochs = epochs
        self.requests: list[FoldRequest] = []

    def __call__(self, request: FoldRequest) -> FoldOutcome:
        self.requests.append(request)
        # Вердикты по умолчанию разные: константа означала бы схлопывание, и
        # тест «обычного» прогона молча проверял бы вырожденный случай.
        varied = [(0.9, 0.9), (0.1, 0.9), (0.9, 0.1), (0.1, 0.1)]
        pairs = self.probs or [varied[index % 4] for index in range(len(request.held_out))]
        probs = [AxisProbs(p_faith=p_f, p_rel=p_r) for p_f, p_r in pairs]
        for epoch in range(1, self.epochs + 1):
            request.on_epoch_end(epoch, probs)
        return FoldOutcome(probs=tuple(probs), checkpoint="results/ft_judge/test/epoch3")


# --------------------------------------------------------------------------- #
# Формат обучающих примеров
# --------------------------------------------------------------------------- #


def test_build_examples_gives_two_examples_per_case() -> None:
    samples = make_corpus(4)
    examples = build_examples(samples)
    assert len(examples) == 2 * len(samples)
    assert {example.axis for example in examples} == {"faithfulness", "relevance"}


def test_example_labels_come_from_the_matching_axis() -> None:
    sample = make_sample("case_x", faithfulness=1, relevance=0)
    by_axis = {example.axis: example for example in build_examples([sample])}
    assert by_axis["faithfulness"].label == 1
    assert by_axis["relevance"].label == 0


def test_completion_parses_back_to_the_same_verdict() -> None:
    """Симметрия формата: обучающее завершение читается парсером инференса."""
    for label in (0, 1):
        example = JudgeExample(
            sample_id="case_x",
            axis="faithfulness",
            system="... FAITHFULNESS: ...",
            user="",
            completion=completion_text("faithfulness", label),
            label=label,
        )
        check_format_symmetry(example)


def test_symmetry_check_rejects_a_completion_that_contradicts_the_label() -> None:
    example = JudgeExample(
        sample_id="case_x",
        axis="faithfulness",
        system="... FAITHFULNESS: ...",
        user="",
        completion="FAITHFULNESS: FAIL",
        label=1,
    )
    with pytest.raises(ValueError, match="parses back to 0"):
        check_format_symmetry(example)


def test_symmetry_check_rejects_a_prompt_without_the_axis_anchor() -> None:
    example = JudgeExample(
        sample_id="case_x",
        axis="relevance",
        system="оцени ответ",
        user="",
        completion="RELEVANCE: PASS",
        label=1,
    )
    with pytest.raises(ValueError, match="never asks for the RELEVANCE"):
        check_format_symmetry(example)


def test_marker_mode_completion_carries_a_marker_line() -> None:
    sample = make_sample("case_x", faithfulness=0)
    examples = build_examples([sample], mode="marker")
    faith = next(example for example in examples if example.axis == "faithfulness")
    assert faith.completion.startswith("MARKER: unsupported_claim")
    check_format_symmetry(faith, mode="marker")


def test_direct_mode_completion_has_no_marker_line() -> None:
    examples = build_examples([make_sample("case_x")], mode="direct")
    assert all("MARKER" not in example.completion for example in examples)


def test_gold_marker_is_none_for_a_passing_axis() -> None:
    """Маркер описывает дефект целиком: приписывать его прошедшей оси нельзя."""
    sample = make_sample("case_x", faithfulness=0)
    assert gold_marker(sample, label=1) == "none"
    assert gold_marker(sample, label=0) == "unsupported_claim"


def test_unlabelled_defect_falls_back_to_unknown() -> None:
    sample = RagSample(
        id="case_x", question="q", context="c", answer="a", faithfulness=0, relevance=1
    )
    assert gold_marker(sample, label=0) == "unknown"


def test_completion_text_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        completion_text("faithfulness", 1, mode="whatever")


# --------------------------------------------------------------------------- #
# Дисбаланс классов — то, из-за чего прогон схлопывался
# --------------------------------------------------------------------------- #


def test_class_balance_counts_each_axis_separately() -> None:
    samples = make_corpus(12)
    balance = class_balance(build_examples(samples))
    assert balance["faithfulness"] == {"pass": 9, "fail": 3}
    assert balance["relevance"] == {"pass": 8, "fail": 4}


def test_pos_weight_downweights_the_dominant_class() -> None:
    # 72/28 — фактический перекос корпуса кураторов.
    labels = [1] * 72 + [0] * 28
    assert compute_pos_weight(labels, "balanced") == pytest.approx(28 / 72)
    assert compute_pos_weight(labels, "none") == 1.0


def test_pos_weight_survives_a_fold_without_positives() -> None:
    assert compute_pos_weight([0, 0, 0], "balanced") == 3.0


def test_oversampling_brings_negatives_up_to_parity() -> None:
    examples = build_examples(make_corpus(20))
    balanced = oversample_negatives(examples, seed=0)
    counts = class_balance(balanced)
    for axis, item in counts.items():
        assert item["fail"] == item["pass"], axis


def test_oversampling_is_deterministic_under_the_same_seed() -> None:
    examples = build_examples(make_corpus(20))
    first = [example.sample_id for example in oversample_negatives(examples, seed=7)]
    second = [example.sample_id for example in oversample_negatives(examples, seed=7)]
    assert first == second


def test_oversampling_leaves_a_balanced_axis_alone() -> None:
    samples = [make_sample(f"case_{i}", faithfulness=i % 2) for i in range(10)]
    examples = [e for e in build_examples(samples) if e.axis == "faithfulness"]
    assert len(oversample_negatives(examples, seed=0)) == len(examples)


# --------------------------------------------------------------------------- #
# Раскладка по фолдам
# --------------------------------------------------------------------------- #


def test_fold_partition_holds_out_exactly_its_own_fold() -> None:
    samples = make_corpus(20)
    folds = make_folds(samples)
    train, held_out = fold_partition(samples, folds, repeat=0, fold=2)
    assert {sample.id for sample in train} & {sample.id for sample in held_out} == set()
    assert all(folds.assignment[sample.id][0] == 2 for sample in held_out)
    assert all(folds.assignment[sample.id][0] != 2 for sample in train)


def test_cases_without_a_fold_assignment_are_used_nowhere() -> None:
    """Кейсы вне folds.json — одна oversized-группа; их нельзя предсказать OOF."""
    samples = make_corpus(20)
    folds = make_folds(samples)
    orphan = make_sample("case_orphan")
    train, held_out = fold_partition([*samples, orphan], folds, repeat=0, fold=1)
    assert orphan.id not in {sample.id for sample in [*train, *held_out]}


def test_fold_partition_rejects_a_repeat_the_folds_file_does_not_have() -> None:
    samples = make_corpus(10)
    folds = make_folds(samples, n_repeats=2)
    with pytest.raises(ValueError, match=r"repeat must be in \[0, 2\)"):
        fold_partition(samples, folds, repeat=5, fold=0)


def test_fold_partition_rejects_an_out_of_range_fold() -> None:
    samples = make_corpus(10)
    folds = make_folds(samples, n_folds=5)
    with pytest.raises(ValueError, match=r"fold must be in \[0, 5\)"):
        fold_partition(samples, folds, fold=9, repeat=0)


# --------------------------------------------------------------------------- #
# Обучение фолда: контракт с тренером
# --------------------------------------------------------------------------- #


def test_train_one_fold_scores_exactly_its_held_out_part() -> None:
    samples = make_corpus(20)
    folds = make_folds(samples)
    trainer = RecordingTrainer()
    result = train_one_fold(samples, folds, FtConfig(), fold=0, train_fold=trainer)

    _, held_out = fold_partition(samples, folds, repeat=0, fold=0)
    assert set(result.probs) == {sample.id for sample in held_out}


def test_training_part_never_contains_a_held_out_case() -> None:
    """Изоляция фолда — то, ради чего сплит и читается из folds.json."""
    samples = make_corpus(20)
    folds = make_folds(samples)
    trainer = RecordingTrainer()
    train_one_fold(samples, folds, FtConfig(), fold=3, train_fold=trainer)

    request = trainer.requests[0]
    held_out_ids = {sample.id for sample in request.held_out}
    assert {example.sample_id for example in request.train_examples} & held_out_ids == set()


def test_trainer_gets_the_pos_weight_computed_on_its_own_fold() -> None:
    samples = make_corpus(20)
    folds = make_folds(samples)
    trainer = RecordingTrainer()
    result = train_one_fold(samples, folds, FtConfig(), fold=0, train_fold=trainer)
    assert trainer.requests[0].pos_weight == pytest.approx(result.pos_weight)
    assert 0.0 < result.pos_weight < 1.0


def test_missing_epoch_diagnostics_is_an_error() -> None:
    """degenerate_rate после каждой эпохи обязателен, а не желателен."""
    samples = make_corpus(20)
    folds = make_folds(samples)

    def silent(request: FoldRequest) -> FoldOutcome:
        probs = [AxisProbs(p_faith=0.7, p_rel=0.7)] * len(request.held_out)
        request.on_epoch_end(1, probs)
        return FoldOutcome(probs=tuple(probs))

    with pytest.raises(ValueError, match="collapse diagnostics ran 1 time"):
        train_one_fold(samples, folds, FtConfig(epochs=3), fold=0, train_fold=silent)


def test_resume_does_not_demand_diagnostics_for_epochs_it_did_not_train() -> None:
    samples = make_corpus(20)
    folds = make_folds(samples)

    def resumed(request: FoldRequest) -> FoldOutcome:
        assert request.resume is True
        probs = [AxisProbs(p_faith=0.7, p_rel=0.3)] * len(request.held_out)
        request.on_epoch_end(3, probs)
        return FoldOutcome(probs=tuple(probs))

    result = train_one_fold(
        samples, folds, FtConfig(epochs=3), fold=0, resume=True, train_fold=resumed
    )
    assert len(result.epochs) == 1


def test_a_trainer_that_scores_the_wrong_number_of_cases_is_an_error() -> None:
    samples = make_corpus(20)
    folds = make_folds(samples)

    def short(request: FoldRequest) -> FoldOutcome:
        probs = [AxisProbs(p_faith=0.6, p_rel=0.6)] * (len(request.held_out) - 1)
        for epoch in range(1, 4):
            request.on_epoch_end(epoch, [*probs, AxisProbs(p_faith=0.6, p_rel=0.6)])
        return FoldOutcome(probs=tuple(probs))

    with pytest.raises(ValueError, match="probability pair"):
        train_one_fold(samples, folds, FtConfig(), fold=0, train_fold=short)


def test_an_empty_fold_is_an_error_rather_than_an_empty_artifact() -> None:
    samples = make_corpus(20)
    folds = make_folds(samples)
    with pytest.raises(ValueError, match="no held-out case"):
        train_one_fold(samples[:1], folds, FtConfig(), fold=4, train_fold=RecordingTrainer())


# --------------------------------------------------------------------------- #
# Схлопывание
# --------------------------------------------------------------------------- #


def test_constant_verdict_is_reported_as_collapsed() -> None:
    """Прогон на 1.5B схлопнулся в (1,1) и прошёл незамеченным — больше нет."""
    samples = make_corpus(20)
    folds = make_folds(samples)
    trainer = RecordingTrainer(probs=[(0.99, 0.99)] * 4)
    result = train_one_fold(samples, folds, FtConfig(), fold=0, train_fold=trainer)
    assert result.collapsed
    assert result.collapse_reason == "pooled"
    assert result.diagnostics()["const_share"] == 1.0


def test_a_varied_verdict_is_not_collapsed() -> None:
    samples = make_corpus(20)
    folds = make_folds(samples)
    trainer = RecordingTrainer(probs=[(0.9, 0.9), (0.1, 0.9), (0.9, 0.1), (0.1, 0.1)])
    result = train_one_fold(samples, folds, FtConfig(), fold=0, train_fold=trainer)
    assert not result.collapsed
    assert result.collapse_reason is None


def test_collapse_on_the_last_epoch_is_caught_even_when_the_pool_looks_fine() -> None:
    """Усреднение по эпохам смазало бы схлопывание, случившееся к концу обучения."""
    result = FoldResult(
        probs={
            "a": AxisProbs(p_faith=0.9, p_rel=0.9),
            "b": AxisProbs(p_faith=0.1, p_rel=0.1),
        }
    )
    from rag_reliability.methods.ft_judge.train import make_epoch_hook  # noqa: PLC0415

    hook = make_epoch_hook(repeat=0, fold=0, sink=result.epochs)
    hook(1, [AxisProbs(p_faith=0.9, p_rel=0.9), AxisProbs(p_faith=0.1, p_rel=0.1)])
    hook(2, [AxisProbs(p_faith=0.9, p_rel=0.9)] * 2)
    assert result.collapse_reason == "last_epoch"


def test_diagnostics_log_every_epoch_in_the_encoder_shape() -> None:
    samples = make_corpus(20)
    folds = make_folds(samples)
    result = train_one_fold(
        samples, folds, FtConfig(epochs=3), fold=0, train_fold=RecordingTrainer()
    )
    diagnostics = result.diagnostics()
    assert [log["epoch"] for log in diagnostics["epochs"]] == [1, 2, 3]
    for log in diagnostics["epochs"]:
        assert set(log) >= {"epoch", "const_share", "output_entropy", "is_degenerate"}


# --------------------------------------------------------------------------- #
# Артефакт
# --------------------------------------------------------------------------- #


def test_artifact_carries_probabilities_and_no_decision() -> None:
    """Бинаризация — дело протокола; метод отдаёт вероятности."""
    predictions = probs_to_predictions({"a": AxisProbs(p_faith=0.7, p_rel=0.2)})
    assert predictions[0].scores == {FAITH_KEY: 0.7, REL_KEY: 0.2}
    assert predictions[0].faithfulness_pred == 0
    assert predictions[0].relevance_pred == 0


def test_probabilities_outside_the_unit_interval_are_rejected() -> None:
    with pytest.raises(ValueError, match="probability in"):
        AxisProbs(p_faith=1.4, p_rel=0.5)


def test_diagnostic_decisions_use_both_axes() -> None:
    decisions = decisions_from_probs({"a": AxisProbs(p_faith=0.9, p_rel=0.1)})
    assert (decisions[0].faithfulness_pred, decisions[0].relevance_pred) == (1, 0)


def test_an_empty_artifact_is_an_error() -> None:
    with pytest.raises(ValueError, match="empty set of probabilities"):
        probs_to_predictions({})


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def write_corpus(path: Path, samples: list[RagSample]) -> None:
    path.write_text(
        "\n".join(json.dumps(sample.model_dump(), ensure_ascii=False) for sample in samples)
        + "\n",
        encoding="utf-8",
    )


def write_folds(path: Path, folds: Folds) -> None:
    path.write_text(
        json.dumps(
            {
                "config": {"n_folds": folds.n_folds, "n_repeats": folds.n_repeats},
                "corpus": {"n": folds.corpus_n, "sha256": folds.sha256},
                "assignment": folds.assignment,
            }
        ),
        encoding="utf-8",
    )


@pytest.fixture
def cli(tmp_path: Path) -> dict:
    samples = make_corpus(20)
    folds = make_folds(samples)
    data = tmp_path / "corpus.jsonl"
    folds_path = tmp_path / "folds.json"
    write_corpus(data, samples)
    write_folds(folds_path, folds)
    return {
        "data": data,
        "folds": folds_path,
        "scores": tmp_path / "scores.jsonl",
        "diagnostics": tmp_path / "ft_diagnostics.json",
        "samples": samples,
    }


def run_cli(cli: dict, *extra: str, trainer: RecordingTrainer | None = None) -> int:
    return train_ft_judge.main(
        [
            "--data",
            str(cli["data"]),
            "--folds",
            str(cli["folds"]),
            "--fold",
            "0",
            "--predictions-output",
            str(cli["scores"]),
            "--diagnostics-output",
            str(cli["diagnostics"]),
            *extra,
        ],
        train_fold=trainer or RecordingTrainer(),
    )


def test_cli_writes_scores_diagnostics_and_run_yaml(cli: dict) -> None:
    assert run_cli(cli) == 0
    rows = [json.loads(line) for line in cli["scores"].read_text(encoding="utf-8").splitlines()]
    assert rows and all(FAITH_KEY in row["scores"] for row in rows)

    diagnostics = json.loads(cli["diagnostics"].read_text(encoding="utf-8"))
    assert diagnostics["fold"] == 0
    assert len(diagnostics["epochs"]) == 3

    run_yaml = yaml.safe_load((cli["scores"].parent / "run.yaml").read_text(encoding="utf-8"))
    assert run_yaml["git"]["hash"] is not None or run_yaml["git"]["dirty"]
    assert run_yaml["ft_judge"]["collapsed"] is False
    assert run_yaml["ft_judge"]["pos_weight"] < 1.0
    assert run_yaml["coverage"]["fold"] == 0


def test_cli_run_yaml_records_collapse_as_a_boolean(cli: dict) -> None:
    """``collapsed`` обязан читаться однозначно, а не строкой argparse."""
    trainer = RecordingTrainer(probs=[(0.99, 0.99)] * 4)
    assert run_cli(cli, trainer=trainer) == 0
    run_yaml = yaml.safe_load((cli["scores"].parent / "run.yaml").read_text(encoding="utf-8"))
    assert run_yaml["ft_judge"]["collapsed"] is True
    assert run_yaml["ft_judge"]["collapse_reason"] == "pooled"


def test_smoke_only_checks_the_format_without_loading_a_model(cli: dict, capsys) -> None:
    assert run_cli(cli, "--smoke-only", "--limit", "5") == 0
    out = capsys.readouterr().out
    assert "Format symmetry: OK" in out
    assert "pos_weight" in out
    assert not cli["scores"].exists()


def test_cli_defaults_the_variant_to_mode_and_fold() -> None:
    args = train_ft_judge.parse_args(["--fold", "3", "--mode", "marker"])
    assert train_ft_judge.variant_name(args) == "marker_fold3"


# --------------------------------------------------------------------------- #
# Реестр
# --------------------------------------------------------------------------- #


def test_ft_judge_is_registered_with_the_judge_score_keys() -> None:
    spec = registry.get("ft_judge")
    assert spec.score_keys == ("m3.p_faith", "m3.p_rel")
    assert spec.corpus_wide is False


def test_ft_judge_cannot_be_scored_case_by_case() -> None:
    """У обучения по фолдам нет покейсового скорера — отказ должен быть внятным."""
    ctx = registry.CommandContext(
        data=Path("data/alfa.jsonl"),
        run_dir=Path("results"),
        predictions_path=Path("results/scores.jsonl"),
    )
    with pytest.raises(ValueError, match="train_ft_judge.py"):
        registry.build_scorer("ft_judge", ctx)


def test_config_rejects_an_unknown_mode() -> None:
    with pytest.raises(ValueError, match="mode must be one of"):
        FtConfig(mode="verdict")


def test_config_rejects_a_save_limit_below_one() -> None:
    with pytest.raises(ValueError, match="save_total_limit"):
        FtConfig(save_total_limit=0)
