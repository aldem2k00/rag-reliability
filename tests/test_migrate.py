"""Tests for one-time migration of legacy prediction artifacts."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml

from scripts.migrate_predictions import (
    discover_runs,
    main,
    migrate_predictions,
)


def _write_predictions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(row) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def _legacy_rows(start: int, count: int) -> list[dict]:
    return [
        {
            "id": f"sample-{index}",
            "p_faith": index / 100,
            "p_rel": (100 - index) / 100,
            "meta": {"source": "fixture"},
        }
        for index in range(start, start + count)
    ]


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_migration_uses_path_specific_prefixes(tmp_path: Path) -> None:
    root = tmp_path / "predictions"
    surface = root / "local" / "baselines" / "surface"
    judge = root / "alfa_openrouter" / "m3" / "zero_shot"
    _write_predictions(surface / "val.jsonl", _legacy_rows(0, 2))
    _write_predictions(judge / "test.jsonl", _legacy_rows(2, 2))

    results = migrate_predictions(root)

    assert [(result.path.relative_to(root).as_posix(), result.prefix) for result in results] == [
        ("alfa_openrouter/m3/zero_shot", "m3"),
        ("local/baselines/surface", "surf"),
    ]
    surface_row = _read_jsonl(surface / "scores.jsonl")[0]
    assert surface_row["scores"] == {
        "surf.p_faith": 0.0,
        "surf.p_rel": 1.0,
    }
    assert surface_row["p_faith"] == 0.0
    assert surface_row["p_rel"] == 1.0
    assert surface_row["meta"] == {"source": "fixture"}
    assert _read_jsonl(judge / "scores.jsonl")[0]["scores"] == {
        "m3.p_faith": 0.02,
        "m3.p_rel": 0.98,
    }


def test_majority_uses_surface_contract_prefix(tmp_path: Path) -> None:
    root = tmp_path / "predictions"
    run_dir = root / "local" / "baselines" / "majority"
    _write_predictions(run_dir / "val.jsonl", _legacy_rows(0, 1))

    [result] = migrate_predictions(root)

    assert result.prefix == "surf"
    assert _read_jsonl(run_dir / "scores.jsonl")[0]["scores"] == {
        "surf.p_faith": 0.0,
        "surf.p_rel": 1.0,
    }


def test_migration_is_idempotent(tmp_path: Path) -> None:
    root = tmp_path / "predictions"
    run_dir = root / "cloud" / "m6"
    _write_predictions(run_dir / "train.jsonl", _legacy_rows(0, 1))
    _write_predictions(run_dir / "val.jsonl", _legacy_rows(1, 1))
    _write_predictions(run_dir / "test.jsonl", _legacy_rows(2, 1))
    existing_metadata = {
        "config": {"m6": {"n_samples": 5}},
        "git_hash": "abc123",
        "seed": 42,
    }
    (run_dir / "run.yaml").write_text(
        yaml.safe_dump(existing_metadata, sort_keys=False),
        encoding="utf-8",
    )

    migrate_predictions(root)
    first_scores = (run_dir / "scores.jsonl").read_bytes()
    first_meta = (run_dir / "run.yaml").read_bytes()
    migrate_predictions(root)

    assert (run_dir / "scores.jsonl").read_bytes() == first_scores
    assert (run_dir / "run.yaml").read_bytes() == first_meta
    assert len(_read_jsonl(run_dir / "scores.jsonl")) == 3
    migrated_metadata = yaml.safe_load(first_meta)
    assert {
        key: migrated_metadata[key] for key in existing_metadata
    } == existing_metadata


def test_partial_migration_records_actual_count(tmp_path: Path) -> None:
    root = tmp_path / "predictions"
    run_dir = root / "cloud" / "m3" / "few_shot"
    _write_predictions(run_dir / "val.jsonl", _legacy_rows(0, 30))

    [result] = migrate_predictions(root)
    metadata = yaml.safe_load((run_dir / "run.yaml").read_text(encoding="utf-8"))

    assert result.n == 30
    assert result.partial is True
    assert metadata["partial"] is True
    assert metadata["n"] == 30


def test_real_surface_artifacts_migrate_without_value_drift(tmp_path: Path) -> None:
    source = Path("predictions/alfa/baselines/surface")
    root = tmp_path / "predictions"
    target = root / "local" / "baselines" / "surface"
    shutil.copytree(source, target)
    original = {
        row["id"]: (row["p_faith"], row["p_rel"])
        for split in ("val", "test")
        for row in _read_jsonl(source / f"{split}.jsonl")
    }

    [result] = migrate_predictions(root)
    migrated = _read_jsonl(target / "scores.jsonl")

    assert result.n == 446
    assert len(migrated) == 446
    for row in migrated:
        expected_faith, expected_rel = original[row["id"]]
        assert row["scores"]["surf.p_faith"] == pytest.approx(expected_faith, abs=1e-12)
        assert row["scores"]["surf.p_rel"] == pytest.approx(expected_rel, abs=1e-12)


def test_dry_run_reads_every_existing_prediction_file_without_writing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    root = Path("predictions")
    split_files = sorted(
        path
        for path in root.rglob("*.jsonl")
        if path.name in {"train.jsonl", "val.jsonl", "test.jsonl"}
    )
    tree_before = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }

    exit_code = main(["--root", str(root), "--dry-run"])
    output = capsys.readouterr().out
    plans = discover_runs(root)

    assert exit_code == 0
    assert len(split_files) >= 22
    assert sum(len(plan.files) for plan in plans) == len(split_files)
    assert "path" in output
    assert "prefix" in output
    assert "partial?" in output
    assert {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    } == tree_before
