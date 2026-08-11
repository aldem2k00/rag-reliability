"""Run provenance: CLI args + git hash + timestamp next to every predictions file."""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
from pathlib import Path
from typing import TypedDict


class GitState(TypedDict):
    hash: str | None
    dirty: bool
    changed: list[str]


def git_short_hash() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, OSError):
        return None


def git_state() -> GitState:
    """Вернуть hash, dirty и изменённые файлы для честной воспроизводимости."""
    try:
        porcelain = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, OSError):
        # Если состояние дерева неизвестно, нельзя утверждать, что запуск был чистым.
        return {"hash": git_short_hash(), "dirty": True, "changed": []}

    records = porcelain.split("\0")
    changed: list[str] = []
    index = 0
    while index < len(records):
        record = records[index]
        if not record:
            break
        status = record[:2]
        changed.append(record[3:])
        if "R" in status or "C" in status:
            index += 1
            changed.append(records[index])
        index += 1
    return {
        "hash": git_short_hash(),
        "dirty": bool(porcelain),
        "changed": changed[:20],
    }


def write_run_meta(path: str | Path, args: argparse.Namespace) -> None:
    state = git_state()
    payload = {
        "args": {key: str(value) for key, value in sorted(vars(args).items())},
        "git_hash": state["hash"],
        "git_dirty": state["dirty"],
        "git_changed": state["changed"],
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
