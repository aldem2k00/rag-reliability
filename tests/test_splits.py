"""Тесты group-aware разбиения корпуса на фолды."""

from __future__ import annotations

import importlib.util
import json
import random
import string
from pathlib import Path

import pytest

from rag_reliability.dataset import save_jsonl
from rag_reliability.schema import RagSample
from rag_reliability.splits import (
    FoldConfig,
    assign_folds,
    build_groups,
    check_folds,
    compute_stats,
    extract_last_client_turn,
    normalize_query,
    oversized_groups,
    write_folds,
)

# Порог, при котором ребро near-duplicate не может возникнуть: косинус <= 1.
# Нужен, чтобы изолировать остальные уровни склейки в юнит-тестах.
NO_NEAR_DUP = 1.01


def _body(rng: random.Random, length: int = 240) -> str:
    """Длинный псевдослучайный текст: делает диалоги заведомо непохожими."""
    return "".join(rng.choice(string.ascii_lowercase + " ") for _ in range(length))


def make_sample(
    sample_id: str,
    *,
    client_turn: str,
    chunk: str,
    reliable: bool = True,
    body: str = "",
) -> RagSample:
    """Синтетический кейс в формате корпуса организаторов."""
    return RagSample(
        id=sample_id,
        question=f"Ассистент: {body or sample_id}\nКлиент: {client_turn}",
        context=f"[CHUNK 1]\n{chunk}\n[CHUNK 2]\nother {sample_id}\n",
        answer=f"Answer {sample_id}",
        faithfulness=int(reliable),
        relevance=int(reliable),
    )


def make_corpus(
    n: int, *, pos_rate: float = 0.7, seed: int = 0, group_size: int = 1
) -> list[RagSample]:
    """n кейсов в группах по ``group_size``, склеенных общим чанком.

    Реплики и диалоги уникальны, так что группу задаёт только ключ чанка — это
    делает предсказуемым и разбиение, и то, что именно ломает подделка.
    """
    rng = random.Random(seed)
    chunks = [
        f"chunk {index} {_body(rng, 60)}" for index in range((n + group_size - 1) // group_size)
    ]
    return [
        make_sample(
            f"s{index:04d}",
            client_turn=f"вопрос {index} {_body(rng, 60)}",
            chunk=chunks[index // group_size],
            reliable=index % 10 < round(pos_rate * 10),
            body=_body(rng),
        )
        for index in range(n)
    ]


# --------------------------------------------------------------------------- #
# Ключи группировки
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Как  ПОДКЛЮЧИТЬ  кэшбэк?", "как подключить кэшбэк"),
        ("как подключить кэшбэк", "как подключить кэшбэк"),
        ("\tКак,  подключить —  кэшбэк!!! ", "как подключить кэшбэк"),
    ],
)
def test_normalize_query_collapses_case_punctuation_and_whitespace(raw: str, expected: str) -> None:
    assert normalize_query(raw) == expected


def test_normalize_query_is_idempotent() -> None:
    """На идемпотентность опирается сравнение ключей генерации и --check."""
    rng = random.Random(7)
    for _ in range(50):
        raw = "".join(rng.choice("Аа Бб?!,.—0123456789\t\n") for _ in range(40))
        once = normalize_query(raw)
        assert normalize_query(once) == once


def test_extract_last_client_turn_returns_the_last_turn_up_to_the_next_role() -> None:
    dialog = (
        "Ассистент: приветствую\n"
        "Клиент: первый вопрос\n"
        "Ассистент: уточните\n"
        "Клиент: второй вопрос\nпродолжение\n"
        "Ассистент: сейчас посмотрю"
    )
    assert extract_last_client_turn(dialog) == "второй вопрос\nпродолжение"


def test_extract_last_client_turn_handles_a_trailing_client_turn() -> None:
    assert extract_last_client_turn("Ассистент: привет\nКлиент: и всё") == "и всё"


def test_extract_last_client_turn_is_none_without_a_client_turn() -> None:
    assert extract_last_client_turn("Ассистент: привет\nОператор: перевожу") is None


# --------------------------------------------------------------------------- #
# build_groups
# --------------------------------------------------------------------------- #


def test_build_groups_is_transitive_across_edge_sources() -> None:
    """A~B по вопросу, B~C по чанку => A, B, C в одной группе."""
    rng = random.Random(1)
    samples = [
        make_sample("a", client_turn="Как подключить кэшбэк?", chunk="chunk-A", body=_body(rng)),
        make_sample("b", client_turn="как  подключить кэшбэк", chunk="chunk-B", body=_body(rng)),
        make_sample("c", client_turn="совсем другой вопрос", chunk="chunk-B", body=_body(rng)),
        make_sample("d", client_turn="третий вопрос", chunk="chunk-D", body=_body(rng)),
    ]
    groups = build_groups(samples, near_dup_threshold=NO_NEAR_DUP)

    assert groups["a"] == groups["b"] == groups["c"]
    assert groups["d"] != groups["a"]


def test_build_groups_joins_near_duplicate_dialogs_without_shared_keys() -> None:
    rng = random.Random(2)
    body = _body(rng, 400)
    twin = body[:200] + "zz" + body[202:]
    samples = [
        make_sample("a", client_turn="первый вопрос", chunk="chunk-A", body=body),
        make_sample("b", client_turn="другой вопрос", chunk="chunk-B", body=twin),
        make_sample("c", client_turn="третий вопрос", chunk="chunk-C", body=_body(rng, 400)),
    ]
    groups = build_groups(samples, near_dup_threshold=0.90)

    assert groups["a"] == groups["b"]
    assert groups["c"] != groups["a"]


def test_build_groups_ignores_chunk_key_when_disabled() -> None:
    rng = random.Random(3)
    samples = [
        make_sample("a", client_turn="первый вопрос", chunk="shared", body=_body(rng)),
        make_sample("b", client_turn="второй вопрос", chunk="shared", body=_body(rng)),
    ]
    assert (
        build_groups(samples, near_dup_threshold=NO_NEAR_DUP, use_chunk_key=False)["a"]
        != (build_groups(samples, near_dup_threshold=NO_NEAR_DUP, use_chunk_key=False)["b"])
    )
    joined = build_groups(samples, near_dup_threshold=NO_NEAR_DUP, use_chunk_key=True)
    assert joined["a"] == joined["b"]


def test_build_groups_groups_samples_without_a_client_turn_by_chunk() -> None:
    rng = random.Random(4)
    samples = [
        RagSample(
            id="a",
            question=f"Ассистент: {_body(rng)}",
            context="[CHUNK 1]\nshared\n[CHUNK 2]\nx\n",
            answer="a",
            faithfulness=1,
            relevance=1,
        ),
        make_sample("b", client_turn="вопрос", chunk="shared", body=_body(rng)),
        make_sample("c", client_turn="вопрос иной", chunk="other", body=_body(rng)),
    ]
    groups = build_groups(samples, near_dup_threshold=NO_NEAR_DUP)

    assert groups["a"] == groups["b"]
    assert groups["c"] != groups["a"]


# --------------------------------------------------------------------------- #
# assign_folds
# --------------------------------------------------------------------------- #


def test_assign_folds_is_deterministic() -> None:
    samples = make_corpus(200, seed=11)
    groups = build_groups(samples, near_dup_threshold=NO_NEAR_DUP)

    first = assign_folds(samples, groups, seed=2233)
    second = assign_folds(samples, groups, seed=2233)

    assert first == second


def test_assign_folds_reacts_to_the_seed() -> None:
    samples = make_corpus(200, seed=12)
    groups = build_groups(samples, near_dup_threshold=NO_NEAR_DUP)

    assert assign_folds(samples, groups, seed=1) != assign_folds(samples, groups, seed=2)


def test_assign_folds_never_splits_a_group_within_a_repeat() -> None:
    rng = random.Random(13)
    samples: list[RagSample] = []
    for group_index in range(60):
        chunk = f"chunk {group_index} {_body(rng, 60)}"
        for member in range(rng.choice([1, 2, 3, 5])):
            samples.append(
                make_sample(
                    f"g{group_index:03d}m{member}",
                    client_turn=f"вопрос {group_index}.{member} {_body(rng, 60)}",
                    chunk=chunk,
                    reliable=(group_index + member) % 10 < 7,
                    body=_body(rng),
                )
            )
    groups = build_groups(samples, near_dup_threshold=NO_NEAR_DUP)
    assignment = assign_folds(samples, groups, n_folds=5, n_repeats=5)

    for repeat in range(5):
        folds_per_group: dict[str, set[int]] = {}
        for sample_id, folds in assignment.items():
            folds_per_group.setdefault(groups[sample_id], set()).add(folds[repeat])
        assert all(len(folds) == 1 for folds in folds_per_group.values())


def test_assign_folds_keeps_the_base_rate_within_tolerance() -> None:
    """Известный дисбаланс 70/30 должен воспроизводиться в каждом фолде."""
    samples = make_corpus(500, pos_rate=0.7, seed=14)
    groups = build_groups(samples, near_dup_threshold=NO_NEAR_DUP)
    assignment = assign_folds(samples, groups, n_folds=5, n_repeats=5)
    stats = compute_stats(
        samples,
        groups,
        assignment,
        FoldConfig(corpus_path=Path("unused"), near_dup_threshold=NO_NEAR_DUP),
    )

    for rates in stats["pos_rate_by_fold"]:
        for rate in rates:
            assert abs(rate - stats["pos_rate_global"]) <= 0.02


def test_oversized_group_is_excluded_from_the_assignment_and_reported() -> None:
    rng = random.Random(15)
    shared = "chunk-oversized"
    samples = [
        make_sample(
            f"big{index:03d}",
            client_turn=f"вопрос {index} {_body(rng, 60)}",
            chunk=shared,
            body=_body(rng),
        )
        for index in range(40)
    ] + make_corpus(160, seed=16)
    groups = build_groups(samples, near_dup_threshold=NO_NEAR_DUP)
    assignment = assign_folds(samples, groups, n_folds=5, n_repeats=5)
    stats = compute_stats(
        samples,
        groups,
        assignment,
        FoldConfig(corpus_path=Path("unused"), near_dup_threshold=NO_NEAR_DUP),
    )

    oversized = oversized_groups(samples, groups)
    assert len(oversized) == 1
    assert stats["oversized_groups"] == oversized
    assert stats["excluded_ids"] == 40
    assert not any(sample_id.startswith("big") for sample_id in assignment)
    assert len(assignment) == 160


# --------------------------------------------------------------------------- #
# Артефакт и валидация
# --------------------------------------------------------------------------- #


def _materialize(tmp_path: Path, samples: list[RagSample]) -> tuple[Path, Path]:
    corpus = tmp_path / "corpus.jsonl"
    save_jsonl(samples, corpus)
    config = FoldConfig(corpus_path=corpus, near_dup_threshold=NO_NEAR_DUP)
    groups = build_groups(
        samples, near_dup_threshold=config.near_dup_threshold, use_chunk_key=config.use_chunk_key
    )
    assignment = assign_folds(
        samples, groups, n_folds=config.n_folds, n_repeats=config.n_repeats, seed=config.seed
    )
    folds = tmp_path / "folds.json"
    write_folds(folds, samples, groups, assignment, config)
    return corpus, folds


def test_write_folds_matches_the_handoff_contract(tmp_path: Path) -> None:
    samples = make_corpus(300, seed=17, group_size=2)
    corpus, folds = _materialize(tmp_path, samples)
    payload = json.loads(folds.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["corpus"]["n"] == len(samples)
    assert payload["corpus"]["path"] == str(corpus).replace("\\", "/")
    assert set(payload["config"]) == {
        "n_folds",
        "n_repeats",
        "seed",
        "near_dup_threshold",
        "use_chunk_key",
    }
    assert set(payload["stats"]["leak_check"]) == {
        "query_overlap",
        f"near_dup_{NO_NEAR_DUP:g}",
        "chunk1_overlap",
    }
    assert all(len(folds_) == 5 for folds_ in payload["assignment"].values())


def test_check_folds_accepts_a_freshly_written_artifact(tmp_path: Path) -> None:
    samples = make_corpus(300, seed=18, group_size=2)
    _, folds = _materialize(tmp_path, samples)

    report = check_folds(folds, samples)

    assert report["passed"], report["errors"]
    assert report["errors"] == []


def test_check_folds_rejects_a_tampered_assignment(tmp_path: Path) -> None:
    """Перенос одного id в другой фолд расщепляет группу — это должно ловиться."""
    samples = make_corpus(300, seed=19, group_size=2)
    _, folds = _materialize(tmp_path, samples)
    payload = json.loads(folds.read_text(encoding="utf-8"))
    groups = build_groups(samples, near_dup_threshold=NO_NEAR_DUP)
    victim = next(
        sample_id
        for sample_id in payload["assignment"]
        if sum(1 for other in groups.values() if other == groups[sample_id]) > 1
    )
    payload["assignment"][victim][0] = (payload["assignment"][victim][0] + 1) % 5
    folds.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    report = check_folds(folds, samples)

    assert not report["passed"]
    assert any("split across folds" in error for error in report["errors"])


def test_check_folds_rejects_a_changed_corpus(tmp_path: Path) -> None:
    samples = make_corpus(300, seed=20, group_size=2)
    corpus, folds = _materialize(tmp_path, samples)
    corpus.write_text(corpus.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    report = check_folds(folds, samples)

    assert not report["passed"]
    assert any("sha256 mismatch" in error for error in report["errors"])


def test_check_folds_rejects_a_dropped_case(tmp_path: Path) -> None:
    """Кейс не может молча выпасть из разбиения."""
    samples = make_corpus(300, seed=21, group_size=2)
    _, folds = _materialize(tmp_path, samples)
    payload = json.loads(folds.read_text(encoding="utf-8"))
    del payload["assignment"][next(iter(payload["assignment"]))]
    folds.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    report = check_folds(folds, samples)

    assert not report["passed"]
    assert any("assignment covers" in error for error in report["errors"])


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

_SPEC = importlib.util.spec_from_file_location(
    "prepare_splits", Path(__file__).parents[1] / "scripts" / "prepare_splits.py"
)
assert _SPEC is not None and _SPEC.loader is not None
prepare_splits = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(prepare_splits)


def _cli_args(corpus: Path, folds: Path) -> list[str]:
    return [
        "--data",
        str(corpus),
        "--output",
        str(folds),
        "--near-dup-threshold",
        str(NO_NEAR_DUP),
    ]


def test_cli_generates_a_valid_artifact_and_is_reproducible(tmp_path: Path) -> None:
    samples = make_corpus(300, seed=22, group_size=2)
    corpus = tmp_path / "corpus.jsonl"
    save_jsonl(samples, corpus)
    folds = tmp_path / "splits" / "folds.json"

    assert prepare_splits.main(_cli_args(corpus, folds)) == 0
    first = folds.read_bytes()
    assert prepare_splits.main(_cli_args(corpus, folds)) == 0

    assert folds.read_bytes() == first


def test_cli_check_returns_zero_on_a_good_artifact(tmp_path: Path) -> None:
    samples = make_corpus(300, seed=23, group_size=2)
    corpus = tmp_path / "corpus.jsonl"
    save_jsonl(samples, corpus)
    folds = tmp_path / "folds.json"
    prepare_splits.main(_cli_args(corpus, folds))

    assert prepare_splits.main(["--check", "--folds", str(folds), "--data", str(corpus)]) == 0


def test_cli_check_returns_one_on_a_tampered_artifact(tmp_path: Path) -> None:
    samples = make_corpus(300, seed=24, group_size=2)
    corpus = tmp_path / "corpus.jsonl"
    save_jsonl(samples, corpus)
    folds = tmp_path / "folds.json"
    prepare_splits.main(_cli_args(corpus, folds))
    payload = json.loads(folds.read_text(encoding="utf-8"))
    groups = build_groups(samples, near_dup_threshold=NO_NEAR_DUP)
    victim = next(
        sample_id
        for sample_id in payload["assignment"]
        if sum(1 for other in groups.values() if other == groups[sample_id]) > 1
    )
    payload["assignment"][victim][0] = (payload["assignment"][victim][0] + 1) % 5
    folds.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    assert prepare_splits.main(["--check", "--folds", str(folds), "--data", str(corpus)]) == 1


def test_check_folds_hashes_the_data_file_it_was_given(tmp_path: Path) -> None:
    """Перетасованный корпус того же состава не должен проходить проверку.

    sha должна считаться с файла, поданного на вход, а не с пути, записанного
    внутри артефакта: иначе проверка «sha256 корпуса совпадает» ничего не стоит.
    """
    samples = make_corpus(300, seed=25, group_size=2)
    _, folds = _materialize(tmp_path, samples)
    reordered = tmp_path / "reordered.jsonl"
    save_jsonl(list(reversed(samples)), reordered)

    assert check_folds(folds, samples)["passed"]

    report = check_folds(folds, samples, corpus_path=reordered)

    assert not report["passed"]
    assert any("sha256 mismatch" in error for error in report["errors"])


def test_check_folds_rejects_tampered_recorded_stats(tmp_path: Path) -> None:
    """Артефакт не может утверждать про себя числа, которых нет в корпусе."""
    samples = make_corpus(300, seed=26, group_size=2)
    _, folds = _materialize(tmp_path, samples)
    payload = json.loads(folds.read_text(encoding="utf-8"))
    assert payload["stats"]["excluded_ids"] == 0  # oversized-групп тут нет
    payload["stats"]["leak_check"]["query_overlap"] = 0.99
    payload["stats"]["excluded_ids"] = 42
    payload["stats"]["n_groups"] = 1
    folds.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    report = check_folds(folds, samples)

    assert not report["passed"]
    assert any("query_overlap" in error for error in report["errors"])
    assert any("excluded_ids" in error for error in report["errors"])


def test_cli_check_rejects_a_different_data_file(tmp_path: Path) -> None:
    samples = make_corpus(300, seed=27, group_size=2)
    corpus = tmp_path / "corpus.jsonl"
    save_jsonl(samples, corpus)
    folds = tmp_path / "folds.json"
    prepare_splits.main(_cli_args(corpus, folds))
    reordered = tmp_path / "reordered.jsonl"
    save_jsonl(list(reversed(samples)), reordered)

    assert prepare_splits.main(["--check", "--folds", str(folds), "--data", str(reordered)]) == 1
