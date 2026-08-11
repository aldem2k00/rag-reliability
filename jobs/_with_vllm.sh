#!/usr/bin/env bash
# Поднять vLLM на VM задания и выполнить переданную команду.
#
# DataSphere Job исполняется без JupyterLab: сервер судьи некому запустить, кроме
# самого задания. Скрипт — ops-обвязка задания, а не логика эксперимента: он ничего
# не считает и не знает ни про корпус, ни про метрики.
#
#   cmd: bash jobs/_with_vllm.sh python scripts/run_gepa.py ...
set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
LOG="${VLLM_LOG:-vllm.log}"
BASE_URL="http://localhost:8000/v1/models"

# --max-logprobs 25 обязателен: клиент судьи запрашивает top_logprobs=20. Без него
# сервер вернёт вердикт без вероятностей и вся ветка деградирует в regex-парсинг.
vllm serve "$MODEL" --port 8000 --max-model-len 8192 \
    --gpu-memory-utilization 0.85 --max-logprobs 25 > "$LOG" 2>&1 &

for _ in $(seq 1 180); do            # до 15 минут: первая загрузка весов долгая
    if curl -sf "$BASE_URL" > /dev/null; then
        echo "vLLM up: $MODEL"
        break
    fi
    sleep 5
done

if ! curl -sf "$BASE_URL" > /dev/null; then
    echo "vLLM не поднялся за 15 минут; хвост $LOG:" >&2
    tail -n 50 "$LOG" >&2 || true
    exit 1
fi

exec "$@"
