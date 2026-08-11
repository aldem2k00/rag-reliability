#!/usr/bin/env python
"""Канонический корпус кураторов: ``data.zip`` -> ``data/alfa.jsonl`` (2233 кейса).

    python scripts/prepare_alfa_corpus.py --source from_organizators/data/data.zip \\
        --output data/alfa.jsonl

Зачем отдельный конвертер. ``scripts/prepare_data.py`` присваивает кейсу номер
строки (``organizer_000001``) и не дедуплицирует: получается 2245 записей. Все
опубликованные артефакты (``predictions/**``) и все опубликованные числа живут
на контентных id ``alfa_<sha1>``, которых ровно 2233 — это канонический счёт из
README и спецификаций. Пересечение двух схем пусто, поэтому ``folds.json``,
построенный на ``organizers.jsonl``, невозможно применить ни к одному артефакту.

Схема id повторяет ``make_case_id`` из исходного загрузчика кураторов:
``"alfa_" + sha1(full_dialog + "\\x00" + answer)[:12]``, причём хешируются сырые
поля CSV, без ``strip()`` — иначе 24 из 446 id не совпадают с опубликованными.

**Владение.** Файл добавлен задачей B1 вынужденно: без канонического корпуса
критерий «прогнан на трёх реальных scores.jsonl» невыполним. По матрице владения
это территория A1; см. раздел PR «Требуется от других».
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import io
import zipfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rag_reliability.schema import ALLOWED_MARKERS, RagSample

#: Поля контекста в выгрузке кураторов.
CHUNK_COLUMNS = tuple(f"chunk_{index}" for index in range(1, 9))

#: 30 МБ CSV с диалогами: поле легко превышает дефолтный лимит модуля csv.
FIELD_SIZE_LIMIT = 1 << 30


def make_case_id(dialog: str, answer: str) -> str:
    """Контентный id кейса: переживает переупорядочивание и пересборку выгрузки."""
    digest = hashlib.sha1((dialog + "\x00" + answer).encode("utf-8")).hexdigest()
    return "alfa_" + digest[:12]


def read_rows(source: str | Path) -> list[dict[str, str]]:
    """Строки ``data.csv`` из zip-архива или из CSV напрямую."""
    csv.field_size_limit(FIELD_SIZE_LIMIT)
    path = Path(source)
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            names = [
                name
                for name in archive.namelist()
                if name.endswith(".csv") and not name.startswith("__MACOSX")
            ]
            if len(names) != 1:
                raise ValueError(f"Expected exactly one CSV inside {path}, found {names}")
            with archive.open(names[0]) as handle:
                return list(csv.DictReader(io.TextIOWrapper(handle, encoding="utf-8")))
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _label(value: object, *, field: str, row_number: int) -> int:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return 1
    if normalized in {"false", "0"}:
        return 0
    raise ValueError(f"Invalid boolean label {field}={value!r} at row {row_number}")


def _markers(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [text]
    if isinstance(parsed, str):
        return [parsed]
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, str) and item]
    return [text]


def _context(row: dict[str, str]) -> str:
    """Тот же формат ``[CHUNK n]``, что у ``prepare_data.organizer_context``.

    От него зависит группировка по ``chunk_1`` в ``splits.build_groups``.
    """
    chunks = []
    for index, column in enumerate(CHUNK_COLUMNS, start=1):
        chunk = str(row.get(column, "") or "").strip()
        if chunk:
            chunks.append(f"[CHUNK {index}]\n{chunk}")
    return "\n\n".join(chunks)


def convert_row(row: dict[str, str], *, row_number: int) -> RagSample:
    dialog = str(row.get("full_dialog", "") or "")
    answer = str(row.get("answer", "") or "")
    faithfulness = _label(
        row.get("binary_faithfulness"), field="binary_faithfulness", row_number=row_number
    )
    relevance = _label(row.get("binary_relevancy"), field="binary_relevancy", row_number=row_number)
    markers = _markers(row.get("markers"))
    if faithfulness == 1 and relevance == 1:
        marker = "none"
    else:
        marker = markers[0] if markers else "unknown"
    if marker not in ALLOWED_MARKERS:
        marker = "unknown"
    return RagSample(
        id=make_case_id(dialog, answer),
        question=dialog.strip(),
        context=_context(row),
        answer=answer.strip(),
        faithfulness=faithfulness,
        relevance=relevance,
        marker=marker,
    )


@dataclass(frozen=True)
class BuildResult:
    """Что именно схлопнулось при канонизации — печатается, а не умалчивается."""

    samples: list[RagSample]
    n_rows: int
    n_duplicate_ids: int
    conflicting_ids: list[str]


def build_corpus(rows: Sequence[dict[str, str]]) -> BuildResult:
    """Дедупликация по контентному id; при конфликте меток побеждает последняя строка.

    Порядок «последняя побеждает» не произволен: под ним ``evaluate_cv
    --legacy-holdout`` воспроизводит все 10 опубликованных чисел до 1e-6, под
    «первая побеждает» — расходится на ~3e-3 из-за единственного кейса
    ``alfa_b3cc00a7a22d``, размеченного в выгрузке дважды и по-разному.
    """
    by_id: dict[str, RagSample] = {}
    labels: dict[str, set[tuple[int, int]]] = {}
    for row_number, row in enumerate(rows, start=1):
        sample = convert_row(row, row_number=row_number)
        by_id[sample.id] = sample
        labels.setdefault(sample.id, set()).add((sample.faithfulness, sample.relevance))
    return BuildResult(
        samples=list(by_id.values()),
        n_rows=len(rows),
        n_duplicate_ids=len(rows) - len(by_id),
        conflicting_ids=sorted(key for key, variants in labels.items() if len(variants) > 1),
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("from_organizators/data/data.zip"))
    parser.add_argument("--output", type=Path, default=Path("data/alfa.jsonl"))
    parser.add_argument(
        "--limit", type=int, default=None, help="smoke-прогон на первых N строках выгрузки"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    rows = read_rows(args.source)
    if args.limit is not None:
        rows = rows[: args.limit]
    result = build_corpus(rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(sample.model_dump_json() + "\n" for sample in result.samples), encoding="utf-8"
    )

    reliable = sum(sample.reliable for sample in result.samples)
    print(f"corpus -> {args.output}")
    print(f"  rows read          {result.n_rows}")
    print(f"  duplicate rows     {result.n_duplicate_ids} (collapsed by content id)")
    print(f"  conflicting labels {len(result.conflicting_ids)} {result.conflicting_ids[:5]}")
    print(f"  cases written      {len(result.samples)}")
    print(f"  reliable rate      {reliable / len(result.samples):.4f}")
    print(f"  markers            {dict(Counter(s.marker for s in result.samples).most_common(5))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
