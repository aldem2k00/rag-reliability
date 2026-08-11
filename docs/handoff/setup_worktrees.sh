#!/usr/bin/env bash
# Разворачивает worktree-ы под волну задач, чтобы агенты работали параллельно
# и не мешали друг другу.
#
#   ./setup_worktrees.sh 1            # A1..A5
#   ./setup_worktrees.sh 1 --push     # + запушить ветки (нужно для Codex Cloud)
#   ./setup_worktrees.sh 2            # B1, B2
#   ./setup_worktrees.sh 3            # C1..C4
#   ./setup_worktrees.sh 4            # D1..D4
#   ./setup_worktrees.sh clean        # удалить все worktree задач
#
# Запускать из корня репозитория. Базовая ветка — integration.
# Совместимо с bash 3.2 (штатный /bin/bash в macOS): без declare -A и прочего bash 4+.

set -eu

BASE_BRANCH="${BASE_BRANCH:-integration}"
WT_ROOT="${WT_ROOT:-..}"

# Волна -> список задач. Через case, а не ассоциативный массив: bash 3.2.
tasks_for_wave() {
  case "$1" in
    1) echo "A1-splits A2-stats A3-schema A4-logprobs A5-hygiene" ;;
    2) echo "B1-protocol B2-score-cli" ;;
    3) echo "C1-stacking C2-encoder C3-m3-axes C4-m6-grounding" ;;
    4) echo "D1-perchunk D2-gepa D3-notebooks D4-reporting" ;;
    *) return 1 ;;
  esac
}

all_tasks() {
  for w in 1 2 3 4; do tasks_for_wave "$w"; done
}

if [ $# -lt 1 ]; then
  echo "укажи номер волны: 1, 2, 3, 4 или clean" >&2
  exit 1
fi

if [ "$1" = "clean" ]; then
  for t in $(all_tasks); do
    id="${t%%-*}"
    git worktree remove --force "$WT_ROOT/wt-$id" 2>/dev/null || true
    git branch -D "task/$t" 2>/dev/null || true
  done
  git worktree prune
  echo "worktree-ы задач удалены"
  exit 0
fi

WAVE="$1"
PUSH="no"
if [ "${2:-}" = "--push" ]; then PUSH="yes"; fi

TASKS="$(tasks_for_wave "$WAVE")" || {
  echo "неизвестная волна: $WAVE (ожидается 1, 2, 3, 4 или clean)" >&2
  exit 1
}

if ! git rev-parse --verify "$BASE_BRANCH" >/dev/null 2>&1; then
  echo "нет ветки $BASE_BRANCH — сначала создай её (см. PROMPTS.md, шаг W0)" >&2
  exit 1
fi

git fetch origin "$BASE_BRANCH" >/dev/null 2>&1 || true

echo "База: $BASE_BRANCH"
echo

CREATED=""
for t in $TASKS; do
  id="${t%%-*}"
  dir="$WT_ROOT/wt-$id"
  if [ -d "$dir" ]; then
    echo "  $id  уже существует: $dir  (пропуск)"
    continue
  fi
  git worktree add "$dir" -b "task/$t" "$BASE_BRANCH" >/dev/null
  echo "  $id  -> $dir   ветка task/$t"
  CREATED="$CREATED $t"
done

if [ "$PUSH" = "yes" ] && [ -n "$CREATED" ]; then
  echo
  echo "Пушу ветки (нужно для Codex Cloud):"
  for t in $CREATED; do
    git push -u origin "task/$t"
  done
fi

echo
echo "Промпты — docs/handoff/PROMPTS.md. Запуск агента в каталоге задачи:"
echo
for t in $TASKS; do
  id="${t%%-*}"
  echo "  cd $WT_ROOT/wt-$id && claude"
done

echo
echo "После слияния волны:"
echo "  git checkout $BASE_BRANCH && git pull && make check"
echo "  ./setup_worktrees.sh clean && ./setup_worktrees.sh $((WAVE + 1)) --push"