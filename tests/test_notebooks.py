"""Структурные тесты ноутбуков DataSphere: ноутбук — пусковая установка, не код.

Ноутбуки не исполняются (для этого нужна A100), поэтому единственная защита от
регрессий — разбор их исходников. Проверяется ровно то, что уже ломалось:

* ``split_samples`` в ячейке — это протокол A с утечкой 24.9% по вопросу;
  любое число, полученное таким сплитом, несравнимо с числами на ``folds.json``;
* клон ветки ``qwen7b-notebook`` — она устарела относительно рабочей;
* ``SAVE_STRATEGY="no"`` — восьмичасовой прогон теряется целиком при обрыве сессии;
* отсутствие смоука на logprobs — извлечение вероятностей молча вырождается в 0.5;
* бизнес-логика в ячейках — она не попадает ни в ``run.yaml``, ни в git-хэш,
  и воспроизводимость прогона умирает первой.

Разбор через ``nbformat``, если он доступен в окружении, иначе через ``json``:
формат ``.ipynb`` — это JSON, и тянуть ради чтения ячеек лишнюю зависимость в
``[dev]`` незачем.
"""

from __future__ import annotations

import json
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
JOBS_DIR = REPO_ROOT / "jobs"

# Всё, что дольше двух часов, идёт через DataSphere Jobs: VM ноутбука
# останавливается при простое и шестичасовой прогон умирает на середине.
REQUIRED_JOBS = (
    *(f"gepa_{variant}_seed{seed}.yaml" for variant in ("markers", "plain") for seed in (0, 1, 2)),
    "encoder_oof.yaml",
    *(f"ft_judge_fold{fold}.yaml" for fold in range(5)),
)
JOB_REQUIRED_KEYS = frozenset(
    {"name", "desc", "cmd", "env", "inputs", "outputs", "cloud-instance-type"}
)

# Ноутбуки, обязательные по карточке D3.
REQUIRED_NOTEBOOKS = (
    "00_setup.ipynb",
    "10_score_judge.ipynb",
    "20_train_encoder.ipynb",
    "30_finetune_judge.ipynb",
)

# Ячейка вправе импортировать только то, что нужно пусковой установке:
# файловые операции, запуск процессов, опрос железа и готовности vLLM, выгрузка
# чекпоинта. Любой sklearn/transformers/trl/numpy в ячейке означает, что расчёт
# переехал из репозитория в ноутбук.
ALLOWED_IMPORTS = frozenset(
    {"os", "sys", "json", "subprocess", "time", "torch", "psutil", "requests", "huggingface_hub"}
)

_IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([A-Za-z_][\w.]*)", re.MULTILINE)
_DEF_RE = re.compile(r"^\s*(?:async\s+def|def|class)\s", re.MULTILINE)
# Вызов, а не упоминание: комментарий «split_samples не вызываем» — это документация,
# а не протокол A. То же с именем устаревшей ветки: запрещена строка, не слово в тексте.
_SPLIT_SAMPLES_CALL_RE = re.compile(r"\bsplit_samples\s*\(")
_STALE_BRANCH_RE = re.compile(r"""["']qwen7b-notebook["']""")
_BRANCH_RE = re.compile(r'^\s*BRANCH\s*=\s*"integration"\s*(?:#.*)?$', re.MULTILINE)
_SAVE_STRATEGY_RE = re.compile(r'^\s*SAVE_STRATEGY\s*=\s*"(\w+)"', re.MULTILINE)
_FLAG_RE = re.compile(r"--[a-zA-Z0-9][\w-]*")


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
    """Склейка всех code-ячеек: markdown может обсуждать что угодно."""
    return "\n".join(_source(c) for c in _read_cells(path) if c["cell_type"] == "code")


def _notebook_paths() -> list[Path]:
    return sorted(NOTEBOOKS_DIR.glob("*.ipynb"))


@pytest.fixture(params=_notebook_paths(), ids=lambda p: p.name)
def notebook(request: pytest.FixtureRequest) -> Path:
    return request.param


def test_required_notebooks_exist() -> None:
    """Четыре ноутбука карточки D3 на месте."""
    missing = [name for name in REQUIRED_NOTEBOOKS if not (NOTEBOOKS_DIR / name).is_file()]
    assert not missing, f"Missing DataSphere notebook(s): {missing}"


def test_notebooks_do_not_call_split_samples(notebook: Path) -> None:
    """Ни один ноутбук не вызывает split_samples — сплит только из folds.json."""
    code = _code(notebook)
    assert not _SPLIT_SAMPLES_CALL_RE.search(code), (
        f"{notebook.name} вызывает split_samples: это протокол A с утечкой 24.9% по вопросу. "
        "Разбиение читается из data/splits/folds*.json."
    )


def test_notebooks_read_folds_json(notebook: Path) -> None:
    """Ноутбуки, которые обучают или оценивают, передают folds.json в CLI."""
    code = _code(notebook)
    if "scripts/evaluate_cv.py" not in code and "train_encoder_baseline.py" not in code:
        pytest.skip(f"{notebook.name} не запускает обучение или оценку")
    assert "folds" in code, f"{notebook.name} не передаёт --folds в CLI"


def test_notebooks_clone_integration_branch(notebook: Path) -> None:
    """BRANCH == 'integration', не 'qwen7b-notebook'."""
    code = _code(notebook)
    assert _BRANCH_RE.search(code), f'{notebook.name} не задаёт BRANCH = "integration"'
    assert not _STALE_BRANCH_RE.search(code), (
        f"{notebook.name} клонирует ветку qwen7b-notebook — она устарела относительно рабочей"
    )


def test_ft_notebook_saves_checkpoints() -> None:
    """SAVE_STRATEGY != 'no' — иначе многочасовой прогон теряется при обрыве."""
    code = _code(NOTEBOOKS_DIR / "30_finetune_judge.ipynb")
    strategies = _SAVE_STRATEGY_RE.findall(code)
    assert strategies == ["epoch"], (
        f'30_finetune_judge.ipynb задаёт SAVE_STRATEGY = {strategies}, ожидалось ["epoch"]: '
        'при "no" обрыв сессии стоит всего многочасового прогона'
    )


def test_setup_notebook_has_logprob_smoke() -> None:
    """В 00_setup есть ассерт на prob_method == 'logprobs' и на разброс вероятностей."""
    code = _code(NOTEBOOKS_DIR / "00_setup.ipynb")
    assert 'prob_method"] == "logprobs"' in code, (
        "00_setup.ipynb обязан ассертить prob_method == 'logprobs': при разбиении "
        "PASS/FAIL на подтокены извлечение молча вырождается в 0.5 для всех кейсов"
    )
    assert "m3.p_faith" in code, "00_setup.ipynb не проверяет разброс вероятностей смоука"


def test_ft_and_encoder_notebooks_show_collapse_control() -> None:
    """Контроль схлопывания печатается в ячейку, а не прячется в файл."""
    for name, key in (
        ("20_train_encoder.ipynb", "const_share"),
        ("30_finetune_judge.ipynb", "degenerate"),
    ):
        code = _code(NOTEBOOKS_DIR / name)
        assert key in code, f"{name} не выводит контроль схлопывания ({key})"


def test_notebooks_have_no_business_logic(notebook: Path) -> None:
    """Ячейки не определяют функций и импортируют только окружение, не счёт."""
    code = _code(notebook)
    definitions = _DEF_RE.findall(code)
    assert not definitions, (
        f"{notebook.name} определяет функцию или класс в ячейке: логика живёт в "
        "репозитории, иначе она не попадает ни в run.yaml, ни в git-хэш. "
        "Не хватает чего-то в CLI — это баг CLI."
    )
    imported = {match.split(".")[0] for match in _IMPORT_RE.findall(code)}
    forbidden = sorted(imported - ALLOWED_IMPORTS)
    assert not forbidden, (
        f"{notebook.name} импортирует {forbidden} — расчёт переехал в ноутбук. "
        f"Разрешено только окружение: {sorted(ALLOWED_IMPORTS)}"
    )


@pytest.fixture(params=sorted(JOBS_DIR.glob("*.yaml")), ids=lambda p: p.name)
def job(request: pytest.FixtureRequest) -> dict:
    import yaml  # noqa: PLC0415

    return dict(yaml.safe_load(request.param.read_text(encoding="utf-8")))


def test_long_runs_have_job_configs() -> None:
    """Заготовлены конфиги Jobs для всех прогонов длиннее 2 часов."""
    missing = [name for name in REQUIRED_JOBS if not (JOBS_DIR / name).is_file()]
    assert not missing, f"Missing DataSphere Job config(s): {missing}"


def test_job_configs_are_complete(job: dict) -> None:
    """Задание без outputs исполнится и молча выбросит артефакт."""
    missing = sorted(JOB_REQUIRED_KEYS - set(job))
    assert not missing, f"Job {job.get('name', '?')} без обязательных полей: {missing}"
    assert job["cloud-instance-type"] == "g2.1", (
        f"Job {job['name']} на конфигурации {job['cloud-instance-type']}: LLM-инференс и "
        "обучение 7B рассчитаны на g2.1 (1× A100 80 GB)"
    )
    assert job["outputs"], f"Job {job['name']} ничего не выгружает — артефакт умрёт вместе с VM"


def test_job_commands_match_cli_signatures(job: dict) -> None:
    """Флаги задания существуют в argparse скрипта, который оно зовёт.

    Конфиги заданий и CLI писались в параллельных ветках одной волны: пока обе
    не влиты, разойтись они не могут, а после слияния расхождение всплывает
    только на GPU через argparse-ошибку — то есть после выделения инстанса.
    Именно так ``jobs/gepa_*.yaml`` пережили переезд ``run_gepa.py`` на
    ``--output-dir``.
    """
    tokens = shlex.split(job["cmd"])
    if tokens[0] == "bash":  # обёртка jobs/_with_vllm.sh перед самой командой
        tokens = tokens[2:]
    script = REPO_ROOT / tokens[1]
    assert script.is_file(), f"Job {job['name']} зовёт несуществующий {tokens[1]}"

    completed = subprocess.run(  # noqa: S603
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        check=False,
    )
    assert completed.returncode == 0, (
        f"{tokens[1]} --help упал, сверить флаги задания {job['name']} нечем:\n"
        f"{completed.stderr[-500:]}"
    )
    known = set(_FLAG_RE.findall(completed.stdout))
    unknown = sorted({token for token in tokens[2:] if token.startswith("--")} - known)
    assert not unknown, (
        f"Job {job['name']} передаёт {unknown} — таких флагов у {tokens[1]} нет. "
        "Задание упадёт на argparse уже после выделения A100."
    )


def test_job_configs_do_not_call_split_samples(job: dict) -> None:
    """Задания читают folds.json, а не режут корпус заново."""
    assert not _SPLIT_SAMPLES_CALL_RE.search(job["cmd"])
    assert "--folds" in job["cmd"], (
        f"Job {job['name']} не передаёт --folds: разбиение приходит из folds.json"
    )


def test_judge_jobs_start_vllm_with_max_logprobs() -> None:
    """У задания нет JupyterLab: vLLM поднимает оно само, и с --max-logprobs 25."""
    import yaml  # noqa: PLC0415

    wrapper = (JOBS_DIR / "_with_vllm.sh").read_text(encoding="utf-8")
    assert "--max-logprobs 25" in wrapper, (
        "клиент судьи запрашивает top_logprobs=20; без --max-logprobs 25 сервер "
        "вернёт вердикт без вероятностей"
    )
    for name in REQUIRED_JOBS:
        cmd = yaml.safe_load((JOBS_DIR / name).read_text(encoding="utf-8"))["cmd"]
        if "--api-base" not in cmd:
            continue
        assert "_with_vllm.sh" in cmd, (
            f"Job {name} ходит в --api-base, но не поднимает vLLM: в задании некому это сделать"
        )


def test_notebooks_are_committed_without_outputs(notebook: Path) -> None:
    """Ноутбуки в git — без выводов: иначе diff нечитаем, а в выводе живут токены."""
    for cell in _read_cells(notebook):
        if cell["cell_type"] != "code":
            continue
        assert not cell.get("outputs"), f"{notebook.name}: очисти outputs перед коммитом"
        assert not cell.get("execution_count"), (
            f"{notebook.name}: очисти execution_count перед коммитом"
        )
