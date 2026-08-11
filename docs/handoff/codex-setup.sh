#!/usr/bin/env bash
# Setup script для Codex Cloud. Выполняется ДО фазы агента, с доступом в интернет.
# У агента интернета не будет, поэтому всё ставится здесь.
set -euo pipefail

python -m pip install --upgrade pip
pip install -e ".[dev]"
pip install hypothesis          # property-тесты (задача A4)

python - <<'PY'
import numpy, scipy, sklearn, pydantic
print("deps ok:", numpy.__version__, scipy.__version__, sklearn.__version__, pydantic.__version__)
PY

# Зелёная отправная точка: если check красный ДО работы агента,
# он потратит сессию на чужие поломки.
make check
