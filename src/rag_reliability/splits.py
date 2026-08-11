"""Group-aware разбиение корпуса на фолды.

Единственный источник истины по разбиению — ``data/splits/folds.json``.
Инференс о сплитах не знает: разбиение применяется только на этапе оценки.

Зачем группы. Стратифицированный ``dataset.split_samples`` разводит по разным
частям кейсы, которые делят один и тот же вопрос клиента, near-duplicate диалог
или одну статью базы знаний. Под этим протоколом тривиальный 1-NN-запоминатель
на char-TF-IDF даёт 0.6278 macro-F1 — выше любого опубликованного результата
проекта, то есть метрика меряет запоминание, а не обобщение. Группа — это
множество кейсов, которые нельзя разводить по фолдам.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_reliability.schema import RagSample

SCHEMA_VERSION = 1

DEFAULT_N_FOLDS = 5
DEFAULT_N_REPEATS = 5
DEFAULT_SEED = 2233
DEFAULT_NEAR_DUP_THRESHOLD = 0.90

#: Доля корпуса, выше которой группа не разбивается, а исключается из метрики.
OVERSIZED_FRACTION = 0.15

#: Стратификация важнее равенства размеров фолдов.
BALANCE_LAMBDA = 5.0

#: Блок строк для поблочного расчёта косинусов: 2233² пар в память не кладём.
NEAR_DUP_BLOCK = 500

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]", re.UNICODE)

#: Роли, которыми размечен диалог в корпусе организаторов.
_ROLE_LINE = re.compile(r"(?m)^(Клиент|Ассистент|Оператор|AlfaGen)\s*:[ \t]*")

_CHUNK_1 = re.compile(r"\[CHUNK 1\](.*?)(?=\[CHUNK \d+\]|\Z)", re.DOTALL)


# --------------------------------------------------------------------------- #
# Ключи группировки
# --------------------------------------------------------------------------- #


def normalize_query(text: str) -> str:
    """Нормализация реплики клиента для группировки: регистр, пунктуация, пробелы.

    Идемпотентна: повторное применение ничего не меняет — на это опирается
    сравнение ключей, собранных на разных этапах (генерация и ``--check``).
    """
    normalized = unicodedata.normalize("NFKC", text).lower()
    normalized = _PUNCT.sub(" ", normalized)
    return _WS.sub(" ", normalized).strip()


def extract_last_client_turn(full_dialog: str) -> str | None:
    """Последняя реплика «Клиент:» до следующей роли.

    ``None`` — если реплики клиента в диалоге нет вовсе (15 строк корпуса).
    Пустая строка — если реплика есть, но пустая: это разные случаи, и
    группировать по пустому ключу нельзя, поэтому их различает вызывающий.
    """
    roles = list(_ROLE_LINE.finditer(full_dialog))
    for index in range(len(roles) - 1, -1, -1):
        match = roles[index]
        if match.group(1) != "Клиент":
            continue
        end = roles[index + 1].start() if index + 1 < len(roles) else len(full_dialog)
        return full_dialog[match.end() : end].strip()
    return None


def extract_chunk_key(context: str) -> str | None:
    """Нормализованный текст первого retrieved-чанка — прокси статьи базы знаний.

    В корпусе организаторов чанк не является отдельным полем: контекст склеен в
    строку с маркерами ``[CHUNK n]``. Заголовок «Название статьи» проставлен лишь
    у части чанков (682 из 2245), поэтому ключ — весь нормализованный текст блока.
    """
    match = _CHUNK_1.search(context)
    if match is None:
        return None
    key = normalize_query(match.group(1))
    return key or None


def _dialog_text(sample: RagSample) -> str:
    """Диалог кейса.

    В корпусе организаторов отдельного поля ``full_dialog`` нет: диалог целиком
    лежит в ``question`` (реплики «Ассистент:»/«Клиент:»).
    """
    return sample.question


def _query_key(sample: RagSample) -> str | None:
    """Ключ уровня 1: нормализованная последняя реплика клиента."""
    turn = extract_last_client_turn(_dialog_text(sample))
    if turn is None:
        return None
    return normalize_query(turn) or None


# --------------------------------------------------------------------------- #
# Union-find и рёбра
# --------------------------------------------------------------------------- #


class _UnionFind:
    """Система непересекающихся множеств со сжатием путей."""

    def __init__(self, items: Iterable[str]) -> None:
        self._parent: dict[str, str] = {item: item for item in items}

    def find(self, item: str) -> str:
        root = item
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[item] != root:
            self._parent[item], item = root, self._parent[item]
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self._parent[max(left_root, right_root)] = min(left_root, right_root)

    def components(self) -> dict[str, list[str]]:
        groups: dict[str, list[str]] = defaultdict(list)
        for item in self._parent:
            groups[self.find(item)].append(item)
        return {root: sorted(members) for root, members in groups.items()}


def _union_by_key(uf: _UnionFind, keys: dict[str, str | None]) -> None:
    """Соединить кейсы с одинаковым непустым ключом."""
    buckets: dict[str, list[str]] = defaultdict(list)
    for sample_id, key in keys.items():
        if key is not None:
            buckets[key].append(sample_id)
    for members in buckets.values():
        first = members[0]
        for other in members[1:]:
            uf.union(first, other)


def near_duplicate_pairs(
    texts: Sequence[str],
    *,
    threshold: float = DEFAULT_NEAR_DUP_THRESHOLD,
    block: int = NEAR_DUP_BLOCK,
) -> list[tuple[int, int]]:
    """Пары индексов с косинусом char 3-5 gram TF-IDF не ниже порога.

    Матрица считается блоками: полная матрица 2233² не помещается в память
    бюджета фазы 0 и не нужна — интересны только пары выше порога.
    """
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if len(texts) < 2:
        return []

    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), dtype=np.float32)
    matrix = vectorizer.fit_transform(texts)

    pairs: list[tuple[int, int]] = []
    n = matrix.shape[0]
    for start in range(0, n, block):
        stop = min(start + block, n)
        similarity = cosine_similarity(matrix[start:stop], matrix)
        rows, cols = np.nonzero(similarity >= threshold)
        for row, col in zip(rows.tolist(), cols.tolist(), strict=True):
            left = start + row
            if left < col:  # каждая пара ровно один раз, диагональ отброшена
                pairs.append((left, col))
    return pairs


def build_groups(
    samples: Sequence[RagSample],
    *,
    near_dup_threshold: float = DEFAULT_NEAR_DUP_THRESHOLD,
    use_chunk_key: bool = True,
) -> dict[str, str]:
    """``id -> group_key``. Union-find поверх трёх источников рёбер.

    1. одинаковый ``normalize_query`` последней реплики клиента;
    2. ``cosine(char 3-5 gram TF-IDF по диалогу) >= near_dup_threshold``;
    3. одинаковый ``chunk_1`` (если ``use_chunk_key``).

    Кейсы без реплики клиента группируются по (2) и (3).

    ``group_key`` контент-адресуем — sha1 отсортированных id участников. Это
    даёт устойчивость между запусками, на которую опирается детерминированный
    tie-break в :func:`assign_folds`.
    """
    ids = [sample.id for sample in samples]
    duplicates = [key for key, count in _counts(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate sample ids in corpus: {duplicates[:5]}")

    uf = _UnionFind(ids)

    _union_by_key(uf, {sample.id: _query_key(sample) for sample in samples})

    for left, right in near_duplicate_pairs(
        [_dialog_text(s) for s in samples], threshold=near_dup_threshold
    ):
        uf.union(ids[left], ids[right])

    if use_chunk_key:
        _union_by_key(uf, {sample.id: extract_chunk_key(sample.context) for sample in samples})

    assignment: dict[str, str] = {}
    for members in uf.components().values():
        key = "grp_" + hashlib.sha1("\n".join(members).encode()).hexdigest()[:12]
        for member in members:
            assignment[member] = key
    return assignment


def _counts(items: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for item in items:
        counts[item] += 1
    return counts


# --------------------------------------------------------------------------- #
# Раскладка групп по фолдам
# --------------------------------------------------------------------------- #


def oversized_groups(samples: Sequence[RagSample], groups: dict[str, str]) -> list[str]:
    """Группы крупнее ``OVERSIZED_FRACTION`` корпуса.

    Такая группа не разбивается: она целиком уходит в train во всех повторах и
    исключается из метрики. Иначе фолд, которому она досталась, становится
    несравним с остальными.
    """
    limit = OVERSIZED_FRACTION * len(samples)
    sizes = _counts(_require_groups(samples, groups))
    return sorted(key for key, size in sizes.items() if size > limit)


def _require_groups(samples: Sequence[RagSample], groups: dict[str, str]) -> list[str]:
    """Ключи групп всех кейсов; отсутствие ключа — ошибка, не молчаливый дефолт."""
    missing = [sample.id for sample in samples if sample.id not in groups]
    if missing:
        raise ValueError(f"Missing group key for {len(missing)} sample(s): {missing[:5]}")
    return [groups[sample.id] for sample in samples]


def assign_folds(
    samples: Sequence[RagSample],
    groups: dict[str, str],
    *,
    n_folds: int = DEFAULT_N_FOLDS,
    n_repeats: int = DEFAULT_N_REPEATS,
    seed: int = DEFAULT_SEED,
) -> dict[str, list[int]]:
    """``id -> [fold_repeat_0, ..., fold_repeat_{n_repeats-1}]``.

    Greedy bin-packing групп с балансировкой доли позитивов::

        cost(fold) = n_fold / n_target
                   + BALANCE_LAMBDA * |pos_fold - pos_rate_global * n_fold| / n_target

    где ``n_fold``/``pos_fold`` — размер и число позитивов фолда *после* добавления
    группы. Оба члена измерены в одних единицах — «кейсов на долю фолда», поэтому
    ``BALANCE_LAMBDA = 5`` читается буквально: один лишний позитив весит как пять
    лишних кейсов размера, то есть стратификация важнее равенства размеров.

    Порядок групп: по убыванию размера, tie-break по ``sha1(group_key + seed+repeat)``.
    Кейсы oversized-групп в результат не попадают — они исключены из метрики.

    Отклонение от формулы карточки (A1 §1, спека 10_PHASE0 §1.3), где записано
    ``|n_fold - n_target| / n_target + LAMBDA * |pos_rate_fold - pos_rate_global|``.
    Оба члена в этой записи вырождаются, и оба измерены на реальных прогонах:

    * размерный член *убывает* по загрузке, пока фолд не добрал до цели: положить
      кейс в фолд со 100 из 100 стоит 0.01, в пустой — 0.99. Минимизация всегда
      выбирает самый полный фолд, один фолд забирает весь корпус, остальные
      остаются пустыми;
    * член доли позитивов взрывается на почти пустых фолдах: первый же кейс даёт
      фолду долю 0 или 1, то есть штраф ``5 * 0.3 = 1.5``, который навсегда
      перевешивает размерный член. Фолд, не получивший группу в начале, не
      получает её никогда — на синтетике 500 кейсов пятый фолд оставался пустым.

    Под обеими записями приёмочные допуски самой карточки (размер фолда ±5%,
    base rate ±2 п.п.) недостижимы. Схема, вес и порядок сохранены; изменены
    только единицы измерения обоих членов.
    """
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")

    _require_groups(samples, groups)
    excluded = set(oversized_groups(samples, groups))

    members: dict[str, list[RagSample]] = defaultdict(list)
    for sample in samples:
        key = groups[sample.id]
        if key not in excluded:
            members[key].append(sample)

    eligible = sum(len(group) for group in members.values())
    if eligible == 0:
        raise ValueError(
            f"Every group is oversized (> {OVERSIZED_FRACTION:.0%} of {len(samples)} samples); "
            "nothing left to evaluate on"
        )

    # Целевая доля позитивов — глобальная по всему корпусу, включая исключённые
    # кейсы: приёмка сравнивает фолды именно с ней.
    pos_rate_global = sum(sample.reliable for sample in samples) / len(samples)
    n_target = eligible / n_folds

    assignment: dict[str, list[int]] = {
        sample.id: [] for group in members.values() for sample in group
    }
    for repeat in range(n_repeats):
        order = sorted(
            members,
            key=lambda key: (-len(members[key]), _tie_break(key, seed + repeat)),
        )
        sizes = [0] * n_folds
        positives = [0] * n_folds
        for key in order:
            group = members[key]
            size = len(group)
            wins = sum(sample.reliable for sample in group)
            fold = min(
                range(n_folds),
                key=lambda index: _cost(
                    sizes[index] + size,
                    positives[index] + wins,
                    n_target,
                    pos_rate_global,
                ),
            )
            sizes[fold] += size
            positives[fold] += wins
            for sample in group:
                assignment[sample.id].append(fold)
    return assignment


def _tie_break(group_key: str, salt: int) -> str:
    return hashlib.sha1(f"{group_key}{salt}".encode()).hexdigest()


def _cost(size: int, positives: int, n_target: float, pos_rate_global: float) -> float:
    """Невязка фолда после добавления группы: загрузка + недобор/перебор позитивов.

    Обе величины — в кейсах на долю фолда, см. :func:`assign_folds`.
    """
    return (size + BALANCE_LAMBDA * abs(positives - pos_rate_global * size)) / n_target


# --------------------------------------------------------------------------- #
# Артефакт folds.json
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class FoldConfig:
    """Параметры протокола разбиения, попадающие в ``folds.json``."""

    corpus_path: Path
    n_folds: int = DEFAULT_N_FOLDS
    n_repeats: int = DEFAULT_N_REPEATS
    seed: int = DEFAULT_SEED
    near_dup_threshold: float = DEFAULT_NEAR_DUP_THRESHOLD
    use_chunk_key: bool = True

    def as_json(self) -> dict[str, Any]:
        return {
            "n_folds": self.n_folds,
            "n_repeats": self.n_repeats,
            "seed": self.seed,
            "near_dup_threshold": self.near_dup_threshold,
            "use_chunk_key": self.use_chunk_key,
        }


def sha256_file(path: str | Path) -> str:
    """sha256 файла корпуса: числа с разных корпусов несравнимы."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _leak_rate(
    keys: dict[str, str | None],
    assignment: dict[str, list[int]],
    repeat: int,
) -> float:
    """Доля назначенных кейсов, делящих ключ с кейсом из другого фолда."""
    if not assignment:
        return 0.0
    folds_by_key: dict[str, set[int]] = defaultdict(set)
    for sample_id, fold_per_repeat in assignment.items():
        key = keys[sample_id]
        if key is not None:
            folds_by_key[key].add(fold_per_repeat[repeat])
    leaked = sum(
        1
        for sample_id in assignment
        if keys[sample_id] is not None and len(folds_by_key[keys[sample_id]]) > 1
    )
    return leaked / len(assignment)


def _near_dup_leak_rate(
    pairs: Sequence[tuple[str, str]],
    assignment: dict[str, list[int]],
    repeat: int,
) -> float:
    """Доля назначенных кейсов, у которых near-duplicate лежит в другом фолде."""
    if not assignment:
        return 0.0
    leaked: set[str] = set()
    for left, right in pairs:
        if left not in assignment or right not in assignment:
            continue
        if assignment[left][repeat] != assignment[right][repeat]:
            leaked.update((left, right))
    return len(leaked) / len(assignment)


def _fold_stats(
    samples: Sequence[RagSample],
    assignment: dict[str, list[int]],
    n_folds: int,
    n_repeats: int,
) -> tuple[list[list[float]], list[list[int]]]:
    """Доля reliable и размер каждого фолда в каждом повторе."""
    by_id = {sample.id: sample for sample in samples}
    pos_rates: list[list[float]] = []
    sizes: list[list[int]] = []
    for repeat in range(n_repeats):
        counts = [0] * n_folds
        positives = [0] * n_folds
        for sample_id, folds in assignment.items():
            fold = folds[repeat]
            counts[fold] += 1
            positives[fold] += by_id[sample_id].reliable
        pos_rates.append([positives[i] / counts[i] if counts[i] else 0.0 for i in range(n_folds)])
        sizes.append(counts)
    return pos_rates, sizes


def compute_stats(
    samples: Sequence[RagSample],
    groups: dict[str, str],
    assignment: dict[str, list[int]],
    config: FoldConfig,
) -> dict[str, Any]:
    """Блок ``stats`` артефакта, включая ``leak_check``.

    Все величины пересчитываются из корпуса, а не переносятся из генерации:
    та же функция обслуживает ``--check``, и записанным числам она не доверяет.
    """
    _require_groups(samples, groups)
    oversized = oversized_groups(samples, groups)
    excluded = [sample.id for sample in samples if groups[sample.id] in set(oversized)]

    query_keys = {sample.id: _query_key(sample) for sample in samples}
    chunk_keys = {sample.id: extract_chunk_key(sample.context) for sample in samples}
    ids = [sample.id for sample in samples]
    pairs = [
        (ids[left], ids[right])
        for left, right in near_duplicate_pairs(
            [_dialog_text(s) for s in samples], threshold=config.near_dup_threshold
        )
    ]

    n_repeats = config.n_repeats
    query_leak = _mean(_leak_rate(query_keys, assignment, r) for r in range(n_repeats))
    chunk_leak = _mean(_leak_rate(chunk_keys, assignment, r) for r in range(n_repeats))
    near_dup_leak = _mean(_near_dup_leak_rate(pairs, assignment, r) for r in range(n_repeats))

    pos_rate_by_fold, size_by_fold = _fold_stats(samples, assignment, config.n_folds, n_repeats)
    sizes = _counts([groups[sample.id] for sample in samples])

    return {
        "n_groups": len(sizes),
        "largest_group": max(sizes.values()) if sizes else 0,
        "oversized_groups": oversized,
        "excluded_ids": len(excluded),
        "pos_rate_global": sum(s.reliable for s in samples) / len(samples),
        "pos_rate_by_fold": pos_rate_by_fold,
        "size_by_fold": size_by_fold,
        "leak_check": {
            "query_overlap": query_leak,
            f"near_dup_{config.near_dup_threshold:g}": near_dup_leak,
            "chunk1_overlap": chunk_leak,
        },
    }


def _mean(values: Iterable[float]) -> float:
    collected = list(values)
    return sum(collected) / len(collected) if collected else 0.0


def write_folds(
    path: str | Path,
    samples: Sequence[RagSample],
    groups: dict[str, str],
    assignment: dict[str, list[int]],
    config: FoldConfig,
) -> None:
    """Записать ``folds.json`` по контракту HANDOFF §7.2, включая stats и leak_check."""
    payload = {
        "schema_version": SCHEMA_VERSION,
        "corpus": {
            "path": str(config.corpus_path).replace("\\", "/"),
            "sha256": sha256_file(config.corpus_path),
            "n": len(samples),
        },
        "config": config.as_json(),
        "stats": compute_stats(samples, groups, assignment, config),
        "assignment": {key: assignment[key] for key in sorted(assignment)},
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1, sort_keys=False) + "\n",
        encoding="utf-8",
    )


# --------------------------------------------------------------------------- #
# Валидация
# --------------------------------------------------------------------------- #

#: Блокирующие проверки и допуски. Порядок — как в спецификации 10_PHASE0 §1.5.
POS_RATE_TOLERANCE = 0.02
NEAR_DUP_LEAK_LIMIT = 0.02
CHUNK_LEAK_LIMIT = 0.05
FOLD_SIZE_TOLERANCE = 0.05


def check_folds(
    path: str | Path,
    samples: Sequence[RagSample],
    *,
    corpus_path: str | Path | None = None,
) -> dict[str, Any]:
    """Валидация существующего ``folds.json``.

    Пересчитывает всё из корпуса, а не читает ``stats``: подделанный или
    устаревший артефакт должен падать, а не подтверждаться собственной записью.
    Пересчитанные величины сравниваются с записанными — расхождение блокирует.

    ``corpus_path`` — файл, из которого загружены ``samples``. Его и надо
    хешировать: иначе перетасованный корпус с тем же составом записей проходит
    проверку, потому что sha считается с пути, записанного внутри артефакта, а
    не с того, что реально подан на вход.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []

    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"schema_version is {payload.get('schema_version')!r}, expected {SCHEMA_VERSION}"
        )

    corpus = payload["corpus"]
    recorded_path = Path(corpus["path"])
    hashed_path = Path(corpus_path) if corpus_path is not None else recorded_path
    if not hashed_path.exists():
        errors.append(f"Corpus file not found: {hashed_path}")
    else:
        actual_sha = sha256_file(hashed_path)
        if actual_sha != corpus["sha256"]:
            errors.append(
                f"Corpus sha256 mismatch: folds.json has {corpus['sha256']}, "
                f"{hashed_path} is {actual_sha}"
            )
    corpus_path_for_config = recorded_path
    if corpus["n"] != len(samples):
        errors.append(
            f"Corpus size mismatch: folds.json has {corpus['n']}, data has {len(samples)}"
        )

    config = FoldConfig(corpus_path=corpus_path_for_config, **payload["config"])
    assignment: dict[str, list[int]] = payload["assignment"]

    known = {sample.id for sample in samples}
    unknown = [sample_id for sample_id in assignment if sample_id not in known]
    if unknown:
        errors.append(f"assignment has {len(unknown)} unknown id(s): {unknown[:5]}")
    bad_shape = [
        sample_id for sample_id, folds in assignment.items() if len(folds) != config.n_repeats
    ]
    if bad_shape:
        errors.append(
            f"assignment has {len(bad_shape)} id(s) with != {config.n_repeats} repeats: "
            f"{bad_shape[:5]}"
        )
    out_of_range = [
        sample_id
        for sample_id, folds in assignment.items()
        if any(not 0 <= fold < config.n_folds for fold in folds)
    ]
    if out_of_range:
        errors.append(
            f"assignment has {len(out_of_range)} id(s) outside [0, {config.n_folds}): "
            f"{out_of_range[:5]}"
        )
    if errors:
        return _report(payload, None, errors, warnings)

    groups = build_groups(
        samples,
        near_dup_threshold=config.near_dup_threshold,
        use_chunk_key=config.use_chunk_key,
    )
    excluded = set(oversized_groups(samples, groups))
    expected_ids = {sample.id for sample in samples if groups[sample.id] not in excluded}
    if set(assignment) != expected_ids:
        errors.append(
            f"assignment covers {len(assignment)} id(s), expected {len(expected_ids)} "
            "(oversized groups must be absent, everything else present)"
        )

    for repeat in range(config.n_repeats):
        split = defaultdict(set)
        for sample_id, folds in assignment.items():
            split[groups[sample_id]].add(folds[repeat])
        broken = sorted(key for key, folds in split.items() if len(folds) > 1)
        if broken:
            errors.append(
                f"repeat {repeat}: {len(broken)} group(s) split across folds: {broken[:5]}"
            )

    stats = compute_stats(samples, groups, assignment, config)
    for mismatch in _stats_mismatches(payload.get("stats"), stats):
        errors.append(f"recorded stats disagree with the corpus: {mismatch}")

    leak = stats["leak_check"]
    near_dup_key = f"near_dup_{config.near_dup_threshold:g}"

    if leak["query_overlap"] > 0.0:
        errors.append(f"query_overlap is {leak['query_overlap']:.4%}, must be 0%")
    if leak[near_dup_key] >= NEAR_DUP_LEAK_LIMIT:
        errors.append(
            f"{near_dup_key} is {leak[near_dup_key]:.4%}, must be < {NEAR_DUP_LEAK_LIMIT:.0%}"
        )
    if leak["chunk1_overlap"] >= CHUNK_LEAK_LIMIT:
        warnings.append(
            f"chunk1_overlap is {leak['chunk1_overlap']:.4%}, expected < {CHUNK_LEAK_LIMIT:.0%}"
        )

    pos_rate_global = stats["pos_rate_global"]
    for repeat, rates in enumerate(stats["pos_rate_by_fold"]):
        for fold, rate in enumerate(rates):
            if abs(rate - pos_rate_global) > POS_RATE_TOLERANCE:
                errors.append(
                    f"repeat {repeat} fold {fold}: pos rate {rate:.4f} deviates from "
                    f"{pos_rate_global:.4f} by more than {POS_RATE_TOLERANCE:.0%} p.p."
                )

    target = len(assignment) / config.n_folds
    for repeat, sizes in enumerate(stats["size_by_fold"]):
        for fold, size in enumerate(sizes):
            if abs(size - target) / target > FOLD_SIZE_TOLERANCE:
                warnings.append(
                    f"repeat {repeat} fold {fold}: size {size} deviates from {target:.1f} "
                    f"by more than {FOLD_SIZE_TOLERANCE:.0%}"
                )

    return _report(payload, stats, errors, warnings)


def _stats_mismatches(recorded: Any, recomputed: Any, prefix: str = "stats") -> list[str]:
    """Расхождения записанного блока ``stats`` с пересчитанным из корпуса.

    Без этого сравнения пересчёт бесполезен как защита: валидатор считал бы
    правильные числа, а артефакт мог бы утверждать любые другие — и проходить.
    """
    if recorded is None:
        return [f"{prefix} is missing"]
    if isinstance(recomputed, dict):
        if not isinstance(recorded, dict):
            return [f"{prefix}: expected an object, got {type(recorded).__name__}"]
        mismatches = []
        for key in sorted(set(recorded) | set(recomputed)):
            if key not in recorded:
                mismatches.append(f"{prefix}.{key} is missing")
            elif key not in recomputed:
                mismatches.append(f"{prefix}.{key} is unexpected")
            else:
                mismatches += _stats_mismatches(recorded[key], recomputed[key], f"{prefix}.{key}")
        return mismatches
    if isinstance(recomputed, list):
        if not isinstance(recorded, list) or len(recorded) != len(recomputed):
            return [f"{prefix}: {recorded!r} != {recomputed!r}"]
        return [
            mismatch
            for index, (left, right) in enumerate(zip(recorded, recomputed, strict=True))
            for mismatch in _stats_mismatches(left, right, f"{prefix}[{index}]")
        ]
    if isinstance(recomputed, float) and isinstance(recorded, int | float):
        # Записано тем же кодом, поэтому расхождение допустимо только на уровне
        # округления json; любое содержательное отличие — подделка или устаревание.
        if abs(float(recorded) - recomputed) <= 1e-9:
            return []
    elif recorded == recomputed and isinstance(recorded, type(recomputed)):
        return []
    return [f"{prefix}: recorded {recorded!r}, corpus says {recomputed!r}"]


def _report(
    payload: dict[str, Any],
    stats: dict[str, Any] | None,
    errors: list[str],
    warnings: list[str],
) -> dict[str, Any]:
    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "recomputed_stats": stats,
        "recorded_stats": payload.get("stats"),
    }
