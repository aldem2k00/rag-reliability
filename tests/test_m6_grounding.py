"""Тесты NLI-grounding на мок-NLI: весов и GPU не нужно.

Мок — детерминированная таблица (premise, hypothesis) -> (entail, contra), так
что проверяются свойства агрегации, а не поведение конкретной NLI-модели.
"""

from __future__ import annotations

import math

import pytest

from rag_reliability.methods.m6.grounding import (
    GROUNDING_KEYS,
    compute_grounding,
    grounding_features,
    score_matrix,
)


class TableNLI:
    """Мок-NLI: явная таблица пар, дефолт — «не подкреплено, но и не противоречит»."""

    def __init__(self, table: dict[tuple[str, str], tuple[float, float]]) -> None:
        self.table = table
        self.calls = 0

    def score(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        self.calls += 1
        rows = []
        for pair in pairs:
            entail, contra = self.table.get(pair, (0.05, 0.05))
            rows.append({"entail": entail, "contra": contra})
        return rows


def split_lines(text: str) -> list[str]:
    """Простой сплиттер для тестов: одна строка — одно «предложение»."""
    return [line.strip() for line in text.splitlines() if line.strip()]


SUPPORTED = "Комиссия составляет 100 рублей."
SUPPORTED_2 = "Срок перевода — 5 дней."
UNSUPPORTED = "Кэшбэк начисляется 30 числа."
CHUNK_A = "chunk about комиссия"
CHUNK_B = "chunk about сроки"


def make_nli(**overrides: tuple[float, float]) -> TableNLI:
    table = {
        (CHUNK_A, SUPPORTED): (0.90, 0.02),
        (CHUNK_B, SUPPORTED): (0.10, 0.05),
        (CHUNK_A, SUPPORTED_2): (0.20, 0.05),
        (CHUNK_B, SUPPORTED_2): (0.85, 0.03),
        (CHUNK_A, UNSUPPORTED): (0.05, 0.40),
        (CHUNK_B, UNSUPPORTED): (0.08, 0.60),
    }
    for key, value in overrides.items():
        sentence, chunk = key.split("__")
        table[({"a": CHUNK_A, "b": CHUNK_B}[chunk], sentence)] = value
    return TableNLI(table)


# --------------------------------------------------------------------------- #
# Контракт фич
# --------------------------------------------------------------------------- #


def test_returns_exactly_eight_declared_finite_features() -> None:
    features = grounding_features(
        f"{SUPPORTED}\n{SUPPORTED_2}",
        [CHUNK_A, CHUNK_B],
        make_nli(),
        sentence_splitter=split_lines,
    )

    assert set(features) == set(GROUNDING_KEYS)
    assert len(GROUNDING_KEYS) == 8
    assert all(math.isfinite(value) for value in features.values())
    assert all(key.startswith("m6.") for key in features)


def test_matrix_shape_is_sentences_by_chunks() -> None:
    entail, contra = score_matrix([SUPPORTED, UNSUPPORTED], [CHUNK_A, CHUNK_B], make_nli())

    assert entail.shape == (2, 2)
    assert contra.shape == (2, 2)
    assert entail[0, 0] == pytest.approx(0.90)
    assert contra[1, 1] == pytest.approx(0.60)


def test_one_nli_call_per_case() -> None:
    """Матрица считается одним батчем: покейсовые вызовы — это стоимость ветки."""
    nli = make_nli()

    compute_grounding(
        f"{SUPPORTED}\n{SUPPORTED_2}\n{UNSUPPORTED}",
        [CHUNK_A, CHUNK_B],
        nli,
        sentence_splitter=split_lines,
    )

    assert nli.calls == 1


# --------------------------------------------------------------------------- #
# min_entail — целевая фича
# --------------------------------------------------------------------------- #


def test_min_entail_reacts_to_a_single_unsupported_sentence() -> None:
    """Слабейшее звено: одно неподкреплённое предложение среди подкреплённых."""
    chunks = [CHUNK_A, CHUNK_B]
    all_supported = grounding_features(
        f"{SUPPORTED}\n{SUPPORTED_2}", chunks, make_nli(), sentence_splitter=split_lines
    )
    one_bad = grounding_features(
        f"{SUPPORTED}\n{SUPPORTED_2}\n{UNSUPPORTED}",
        chunks,
        make_nli(),
        sentence_splitter=split_lines,
    )

    assert one_bad["m6.min_entail"] < all_supported["m6.min_entail"]
    assert one_bad["m6.min_entail"] == pytest.approx(0.08)
    # среднее размывает сигнал сильнее, чем минимум, — ради этого фича и введена
    mean_drop = all_supported["m6.mean_entail"] - one_bad["m6.mean_entail"]
    min_drop = all_supported["m6.min_entail"] - one_bad["m6.min_entail"]
    assert min_drop > mean_drop


def test_min_entail_is_monotone_in_the_weakest_sentence() -> None:
    previous = -1.0
    for entail in (0.05, 0.3, 0.6, 0.95):
        nli = make_nli(**{f"{UNSUPPORTED}__a": (entail, 0.1)})
        features = grounding_features(
            f"{SUPPORTED}\n{UNSUPPORTED}", [CHUNK_A, CHUNK_B], nli, sentence_splitter=split_lines
        )
        assert features["m6.min_entail"] > previous
        previous = features["m6.min_entail"]


def test_max_entail_ignores_the_weak_sentence() -> None:
    features = grounding_features(
        f"{SUPPORTED}\n{UNSUPPORTED}", [CHUNK_A, CHUNK_B], make_nli(), sentence_splitter=split_lines
    )

    assert features["m6.max_entail"] == pytest.approx(0.90)


def test_frac_unsupported_counts_sentences_below_threshold() -> None:
    features = grounding_features(
        f"{SUPPORTED}\n{SUPPORTED_2}\n{UNSUPPORTED}",
        [CHUNK_A, CHUNK_B],
        make_nli(),
        sentence_splitter=split_lines,
        entail_threshold=0.5,
    )

    assert features["m6.frac_unsupported"] == pytest.approx(1 / 3)
    assert features["m6.n_sentences"] == pytest.approx(3.0)


def test_contra_features_take_the_worst_chunk() -> None:
    features = grounding_features(
        UNSUPPORTED, [CHUNK_A, CHUNK_B], make_nli(), sentence_splitter=split_lines
    )

    assert features["m6.max_contra"] == pytest.approx(0.60)
    assert features["m6.mean_contra"] == pytest.approx(0.60)


# --------------------------------------------------------------------------- #
# chunk_spread — детектор смешения источников
# --------------------------------------------------------------------------- #


def test_chunk_spread_grows_when_sentences_lean_on_different_chunks() -> None:
    one_source = grounding_features(
        f"{SUPPORTED}\n{SUPPORTED}", [CHUNK_A, CHUNK_B], make_nli(), sentence_splitter=split_lines
    )
    two_sources = grounding_features(
        f"{SUPPORTED}\n{SUPPORTED_2}", [CHUNK_A, CHUNK_B], make_nli(), sentence_splitter=split_lines
    )

    assert one_source["m6.chunk_spread"] == pytest.approx(1.0)
    assert two_sources["m6.chunk_spread"] == pytest.approx(2.0)


def test_source_chunk_ids_follow_argmax_entail() -> None:
    result = compute_grounding(
        f"{SUPPORTED}\n{SUPPORTED_2}",
        [CHUNK_A, CHUNK_B],
        make_nli(),
        sentence_splitter=split_lines,
    )

    assert result.source_chunk_ids == {0, 1}
    assert result.features["m6.chunk_spread"] == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# Границы
# --------------------------------------------------------------------------- #


def test_single_chunk_case_is_valid() -> None:
    features = grounding_features(
        SUPPORTED, [CHUNK_A], make_nli(), sentence_splitter=split_lines
    )

    assert features["m6.chunk_spread"] == pytest.approx(1.0)
    assert features["m6.n_sentences"] == pytest.approx(1.0)


def test_empty_chunks_raise_instead_of_producing_a_perfect_case() -> None:
    with pytest.raises(ValueError, match="no context chunks"):
        grounding_features(SUPPORTED, [], make_nli(), sentence_splitter=split_lines)


def test_blank_answer_still_produces_one_hypothesis() -> None:
    features = grounding_features(
        "   ", [CHUNK_A], make_nli(), sentence_splitter=lambda text: [text]
    )

    assert features["m6.n_sentences"] == pytest.approx(1.0)
    assert all(math.isfinite(value) for value in features.values())


def test_pairs_are_built_premise_chunk_hypothesis_sentence() -> None:
    """Направление пары — часть определения метода, перепутать его нечем поймать."""
    seen: list[tuple[str, str]] = []

    class Recorder:
        def score(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
            seen.extend(pairs)
            return [{"entail": 0.5, "contra": 0.1} for _ in pairs]

    grounding_features(SUPPORTED, [CHUNK_A, CHUNK_B], Recorder(), sentence_splitter=split_lines)

    assert seen == [(CHUNK_A, SUPPORTED), (CHUNK_B, SUPPORTED)]


# --------------------------------------------------------------------------- #
# CLI: scripts/score_m6_grounding.py поверх scripts/score.py
# --------------------------------------------------------------------------- #

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import score_m6_grounding as cli  # noqa: E402

from rag_reliability.dataset import load_jsonl  # noqa: E402
from rag_reliability.methods import registry  # noqa: E402

DUMMY_DATA = "data/dummy.jsonl"


def _run_cli(tmp_path: Path, *extra: str) -> Path:
    output = tmp_path / "scores.jsonl"
    assert cli.main(
        ["--data", DUMMY_DATA, "--output", str(output), "--backend", "dummy", *extra]
    ) == 0
    return output


def test_cli_smoke_writes_twelve_score_keys(tmp_path: Path) -> None:
    output = _run_cli(tmp_path, "--limit", "10")

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    assert len(rows) == 10
    for row in rows:
        assert set(row["scores"]) == set(cli.SCORE_KEYS)
        assert row["faithfulness_pred"] == 0 and row["relevance_pred"] == 0


def test_cli_artifact_passes_the_registry_contract(tmp_path: Path) -> None:
    output = _run_cli(tmp_path, "--limit", "5")

    registry.validate_scores_file(output, cli.SPEC, expected_n=5)


def test_cli_writes_run_yaml_with_pair_counts(tmp_path: Path) -> None:
    _run_cli(tmp_path, "--limit", "5")

    meta = yaml.safe_load((tmp_path / "run.yaml").read_text(encoding="utf-8"))
    assert meta["method"]["name"] == "m6_grounding"
    assert meta["method"]["default_score_expr"] == "m6.min_entail"
    assert meta["partial"] is True
    assert "hash" in meta["git"]


def test_cli_resume_continues_without_duplicates(tmp_path: Path) -> None:
    output = _run_cli(tmp_path, "--limit", "4")

    _run_cli(tmp_path, "--limit", "8", "--resume")

    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines() if line]
    ids = [row["id"] for row in rows]
    assert len(ids) == len(set(ids)) == 8


def test_cli_shuffle_is_deterministic_for_a_seed(tmp_path: Path) -> None:
    samples = load_jsonl(DUMMY_DATA)
    args = cli.parse_args(
        ["--data", DUMMY_DATA, "--output", "x", "--shuffle", "--seed", "7", "--subsample", "5"]
    )

    first = [sample.id for sample in cli.select_samples(samples, args)]
    second = [sample.id for sample in cli.select_samples(samples, args)]

    assert first == second
    assert len(first) == 5
    assert set(first) <= {sample.id for sample in samples}


def test_auc_report_carries_intervals_for_every_feature(tmp_path: Path) -> None:
    output = _run_cli(tmp_path)
    report_path = tmp_path / "auc.json"

    assert cli.main(
        [
            "--auc-only",
            "--data", DUMMY_DATA,
            "--scores", str(output),
            "--auc-report", str(report_path),
            "--bootstrap-B", "200",
        ]
    ) == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["n"] == len(load_jsonl(DUMMY_DATA))
    assert set(report["features"]) == set(cli.SCORE_KEYS)
    for entry in report["features"].values():
        if entry.get("constant"):
            continue
        for target in ("faithfulness", "reliable"):
            assert 0.0 <= entry[target]["auc"] <= 1.0
            assert entry[target]["ci95_lo"] <= entry[target]["auc"] <= entry[target]["ci95_hi"]
