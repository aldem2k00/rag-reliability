"""Migrate legacy split predictions into one namespaced scores artifact."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

EXPECTED_CORPUS_SIZE = 2233
SPLIT_ORDER = ("val", "test", "train")


@dataclass(frozen=True)
class MigrationPlan:
    """One prediction run and the legacy split files that belong to it."""

    path: Path
    prefix: str
    files: tuple[Path, ...]


@dataclass(frozen=True)
class MigrationResult:
    """Summary shared by dry-run output, tests and the write path."""

    path: Path
    prefix: str
    n: int
    partial: bool


def _prefix_for_path(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    parts = relative.parts
    variant = parts[-1]

    if "m3" in parts:
        return "m3"
    if "m6" in parts:
        return "m6"
    if variant.startswith("surface") or variant == "majority":
        return "surf"
    if variant == "curator_encoder":
        return "enc"
    raise ValueError(
        f"Cannot determine score prefix for prediction run {relative.as_posix()!r}"
    )


def discover_runs(root: str | Path) -> list[MigrationPlan]:
    """Find every directory containing at least one canonical split artifact."""
    root = Path(root)
    by_directory: dict[Path, list[Path]] = {}
    for path in root.rglob("*.jsonl"):
        if path.name not in {f"{split}.jsonl" for split in SPLIT_ORDER}:
            continue
        by_directory.setdefault(path.parent, []).append(path)

    plans: list[MigrationPlan] = []
    for directory in sorted(by_directory):
        files_by_name = {path.name: path for path in by_directory[directory]}
        files = tuple(
            files_by_name[f"{split}.jsonl"]
            for split in SPLIT_ORDER
            if f"{split}.jsonl" in files_by_name
        )
        plans.append(
            MigrationPlan(
                path=directory,
                prefix=_prefix_for_path(directory, root),
                files=files,
            )
        )
    return plans


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise TypeError(
                    f"Prediction at {path}:{line_number} must be a JSON object"
                )
            for field in ("id", "p_faith", "p_rel"):
                if field not in row:
                    raise ValueError(
                        f"Missing {field!r} at {path}:{line_number}"
                    )
            rows.append(row)
    return rows


def _migrated_rows(plan: MigrationPlan) -> list[dict]:
    migrated: list[dict] = []
    seen_ids: set[str] = set()
    for path in plan.files:
        for row in _load_rows(path):
            sample_id = str(row["id"])
            if sample_id in seen_ids:
                raise ValueError(
                    f"Duplicate prediction id {sample_id!r} in {plan.path}"
                )
            seen_ids.add(sample_id)

            if "scores" in row:
                existing_scores = row["scores"]
                if not isinstance(existing_scores, dict):
                    raise ValueError(
                        f"Field 'scores' for prediction {sample_id!r} must be an object"
                    )
                scores = dict(existing_scores)
            else:
                scores = {}
            scores[f"{plan.prefix}.p_faith"] = float(row["p_faith"])
            scores[f"{plan.prefix}.p_rel"] = float(row["p_rel"])
            migrated.append({**row, "scores": scores})
    return migrated


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.write_text(payload, encoding="utf-8")


def _update_run_metadata(path: Path, *, n: int, partial: bool) -> None:
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Run metadata at {path} must be a YAML mapping")
        metadata = loaded
    else:
        metadata = {}
    metadata["partial"] = partial
    metadata["n"] = n
    path.write_text(
        yaml.safe_dump(metadata, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def migrate_predictions(
    root: str | Path,
    *,
    dry_run: bool = False,
    expected_n: int = EXPECTED_CORPUS_SIZE,
) -> list[MigrationResult]:
    """Combine legacy splits, preserving old fields and namespacing probabilities."""
    plans = discover_runs(root)
    results: list[MigrationResult] = []
    for plan in plans:
        rows = _migrated_rows(plan)
        result = MigrationResult(
            path=plan.path,
            prefix=plan.prefix,
            n=len(rows),
            partial=len(rows) < expected_n,
        )
        results.append(result)
        if dry_run:
            continue
        _write_jsonl(plan.path / "scores.jsonl", rows)
        _update_run_metadata(
            plan.path / "run.yaml",
            n=result.n,
            partial=result.partial,
        )
    return results


def _print_table(root: Path, results: list[MigrationResult]) -> None:
    headers = ("path", "prefix", "n rows", "partial?")
    rows = [
        (
            result.path.relative_to(root).as_posix(),
            result.prefix,
            str(result.n),
            "yes" if result.partial else "no",
        )
        for result in results
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def render(row: tuple[str, ...]) -> str:
        return " | ".join(
            value.ljust(widths[index]) for index, value in enumerate(row)
        )

    print(render(headers))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(render(row))


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine legacy prediction splits into scores.jsonl."
    )
    parser.add_argument("--root", type=Path, default=Path("predictions"))
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    results = migrate_predictions(args.root, dry_run=args.dry_run)
    _print_table(args.root, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
