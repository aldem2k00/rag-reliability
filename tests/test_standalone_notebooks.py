"""Структурные тесты самостоятельных ноутбуков методов 3 и 6.

Контур ``notebooks/standalone/`` живёт по другим правилам, чем пусковые
``notebooks/*.ipynb``: там ячейка не имеет права считать (``tests/test_notebooks.py``
это проверяет), здесь — обязана. Ноутбук читается как разбор метода, поэтому
промпты, формулы признаков и метрики видны прямо в ячейках, а через CLI идут только
корпусные прогоны, которым нужны ``--resume`` и ``run.yaml`` с git-хэшем.

Ноутбуки не исполняются: метод 3 требует A100 с поднятым vLLM. Проверяется то, что
уже ломалось в этом репозитории и что разбором исходника поймать можно:

* ``split_samples`` в ячейке — протокол с утечкой 24.9% по вопросу; любое число,
  полученное таким сплитом, несравнимо с числами на ``folds.json``;
* отсутствие смоука на logprobs — извлечение вероятностей молча вырождается в 0.5,
  и корпусный прогон выглядит успешным, не неся сигнала;
* пропавшая структура разделов — ноутбук перестаёт читаться как разбор гипотез;
* сохранённые outputs — в них живут токены, а diff становится нечитаем.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
STANDALONE_DIR = REPO_ROOT / "notebooks" / "standalone"

M3 = "method3_judge.ipynb"
M6 = "method6_grounding.ipynb"
REQUIRED_NOTEBOOKS = (M3, M6)

#: Разделы, без которых ноутбук перестаёт быть самостоятельным разбором: откуда
#: берутся данные, чем правится прогон и чем всё кончилось.
REQUIRED_SECTIONS = ("## Config", "## Data", "## Итог")

_SPLIT_SAMPLES_CALL_RE = re.compile(r"\bsplit_samples\s*\(")
_FOLDS_RE = re.compile(r'FOLDS\s*=\s*"data/splits/folds_alfa\.json"')
_BRANCH_RE = re.compile(r'^\s*BRANCH\s*=\s*"(\w+)"', re.MULTILINE)


def _read_cells(path: Path) -> list[dict]:
    """Ячейки ноутбука; nbformat, если он есть, иначе стандартный json."""
    try:
        import nbformat  # noqa: PLC0415
    except ImportError:
        return list(json.loads(path.read_text(encoding="utf-8"))["cells"])
    return list(nbformat.read(str(path), as_version=4).cells)


def _source(cell: dict) -> str:
    source = cell["source"]
    return source if isinstance(source, str) else "".join(source)


def _code(path: Path) -> str:
    return "\n".join(_source(c) for c in _read_cells(path) if c["cell_type"] == "code")


def _markdown(path: Path) -> str:
    return "\n".join(_source(c) for c in _read_cells(path) if c["cell_type"] == "markdown")


@pytest.fixture(params=REQUIRED_NOTEBOOKS)
def notebook(request: pytest.FixtureRequest) -> Path:
    return STANDALONE_DIR / request.param


def test_required_notebooks_exist() -> None:
    missing = [name for name in REQUIRED_NOTEBOOKS if not (STANDALONE_DIR / name).is_file()]
    assert not missing, f"Нет самостоятельного ноутбука: {missing}"


def test_notebook_json_is_valid(notebook: Path) -> None:
    payload = json.loads(notebook.read_text(encoding="utf-8"))
    assert payload["nbformat"] == 4
    assert payload["cells"], f"{notebook.name}: пустой ноутбук"


def test_notebooks_do_not_call_split_samples(notebook: Path) -> None:
    """Разбиение читается из folds_alfa.json, а не нарезается заново."""
    assert not _SPLIT_SAMPLES_CALL_RE.search(_code(notebook)), (
        f"{notebook.name} вызывает split_samples: это сплит с утечкой 24.9% по вопросу, "
        "и полученные им числа несравнимы с остальными"
    )


def test_notebooks_read_the_canonical_folds(notebook: Path) -> None:
    assert _FOLDS_RE.search(_code(notebook)), (
        f'{notebook.name} не задаёт FOLDS = "data/splits/folds_alfa.json" — '
        "единственный источник разбиения"
    )


def test_notebooks_clone_a_real_branch(notebook: Path) -> None:
    """Клонируется актуальная ветка, а не устаревшая qwen7b-notebook."""
    branches = _BRANCH_RE.findall(_code(notebook))
    assert branches == ["main"], (
        f"{notebook.name} задаёт BRANCH = {branches}, ожидалось ['main']: волны 1-4 влиты "
        "в main, прогон с другой ветки несравним с остальными"
    )


def test_notebooks_have_the_required_sections(notebook: Path) -> None:
    markdown = _markdown(notebook)
    missing = [section for section in REQUIRED_SECTIONS if section not in markdown]
    assert not missing, f"{notebook.name}: нет разделов {missing}"


def test_notebooks_state_their_hypotheses(notebook: Path) -> None:
    """Ноутбук идёт по гипотезам, а не по списку команд."""
    markdown = _markdown(notebook)
    assert "ипотез" in markdown, f"{notebook.name} не формулирует ни одной гипотезы"


def test_judge_notebook_has_the_logprob_smoke() -> None:
    """Смоук на logprobs обязателен: без него p_faith молча становится 0.5 у всех."""
    code = _code(STANDALONE_DIR / M3)
    assert 'row["prob_method"] == "logprobs"' in code, (
        f"{M3} обязан ассертить prob_method == 'logprobs': при разбиении PASS/FAIL на "
        "подтокены извлечение вероятностей вырождается в 0.5 для всех кейсов"
    )
    assert "--limit 5" in code, f"{M3}: смоук должен идти на нескольких кейсах, а не на корпусе"


def test_judge_notebook_starts_vllm_with_max_logprobs() -> None:
    code = _code(STANDALONE_DIR / M3)
    assert "--max-logprobs 25" in code, (
        f"{M3}: клиент судьи запрашивает top_logprobs=20; без --max-logprobs 25 сервер "
        "вернёт вердикт без вероятностей"
    )


def test_grounding_notebook_checks_its_formulas_against_the_module() -> None:
    """Формулы признаков в ячейке сверяются с methods/m6/grounding.py.

    Ячейка показывает агрегацию явно — иначе ноутбук не объясняет метод. Расхождение
    с модулем означало бы, что ноутбук объясняет не тот метод, который посчитал числа.
    """
    code = _code(STANDALONE_DIR / M6)
    assert "grounding.features[key]" in code, (
        f"{M6}: формулы в ячейке не сверяются с modules/m6/grounding.py"
    )


def test_notebooks_are_committed_without_outputs(notebook: Path) -> None:
    """Ноутбуки в git — без выводов: в них живут токены, а diff становится нечитаем."""
    for cell in _read_cells(notebook):
        if cell["cell_type"] != "code":
            continue
        assert not cell.get("outputs"), f"{notebook.name}: очисти outputs перед коммитом"
        assert not cell.get("execution_count"), (
            f"{notebook.name}: очисти execution_count перед коммитом"
        )
