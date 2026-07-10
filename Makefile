# Quick commands. Everything runs against .venv (create with `make install`).

PY := .venv/bin/python
DATA ?= data/dummy.jsonl
MODEL ?= mlx-community/Qwen2.5-1.5B-Instruct-4bit
ENCODER_MAX_LENGTH ?= 512
ENCODER_BATCH_SIZE ?= 4
ENCODER_EPOCHS ?= 3
ENCODER_LEARNING_RATE ?= 2e-5
ENCODER_POS_WEIGHT_MODE ?= none

.PHONY: help install install-mlx install-lettucedetect install-m6 install-cloud install-gepa test lint check dummy \
        install-encoder baseline-direct baseline-marker encoder-baseline train-direct \
        train-marker train-lettucedetect infer-direct infer-marker infer-lettucedetect \
        install-demo serve-demo benchmark-dummy eval-all clean

help: ## List available targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

install: ## Create venv and install core + dev deps (uv)
	uv venv --python 3.12
	uv pip install -e ".[dev]"

install-mlx: ## Add MLX backend + LoRA training deps (Apple Silicon)
	uv pip install -e ".[mlx]"

install-lettucedetect: ## Add LettuceDetect feature-classifier deps
	uv pip install -e ".[lettucedetect]"

install-m6: ## Add Method 6 SelfCheck/NLI feature deps
	uv pip install -e ".[m6]"

install-cloud: ## Add OpenAI-compatible cloud backend deps
	uv pip install -e ".[cloud]"

install-gepa: ## Add DSPy deps for GEPA prompt evolution (Method 3)
	uv pip install -e ".[gepa]"

install-encoder: ## Add supervised encoder baseline deps
	uv pip install -e ".[encoder]"

install-demo: ## Add local Gradio demo deps
	uv pip install -e ".[demo]"

test: ## Run unit tests
	$(PY) -m pytest -q

lint: ## Ruff lint
	$(PY) -m ruff check .

check: test lint ## Tests + lint

dummy: ## Smoke-test pipeline without a model (keyword strategy, marker mode)
	$(PY) scripts/run_prompt_baseline.py --data $(DATA) \
		--output results/dummy_marker_predictions.jsonl \
		--mode marker --backend dummy --dummy-strategy keyword
	$(PY) scripts/evaluate.py --data $(DATA) \
		--predictions results/dummy_marker_predictions.jsonl \
		--output results/dummy_marker_metrics.json

baseline-direct: ## Zero-shot MLX baseline, direct mode
	$(PY) scripts/run_prompt_baseline.py --data $(DATA) \
		--output results/qwen_direct_predictions.jsonl \
		--mode direct --backend mlx --model $(MODEL)
	$(PY) scripts/evaluate.py --data $(DATA) \
		--predictions results/qwen_direct_predictions.jsonl \
		--output results/qwen_direct_metrics.json

baseline-marker: ## Zero-shot MLX baseline, marker mode
	$(PY) scripts/run_prompt_baseline.py --data $(DATA) \
		--output results/qwen_marker_predictions.jsonl \
		--mode marker --backend mlx --model $(MODEL)
	$(PY) scripts/evaluate.py --data $(DATA) \
		--predictions results/qwen_marker_predictions.jsonl \
		--output results/qwen_marker_metrics.json

encoder-baseline: ## Supervised RuModernBERT reliability classifier
	$(PY) scripts/train_encoder_baseline.py --data $(DATA) \
		--output results/encoder_baseline_512_best_metrics.json \
		--output-dir results/encoder_checkpoints_512_best \
		--max-length $(ENCODER_MAX_LENGTH) --batch-size $(ENCODER_BATCH_SIZE) \
		--epochs $(ENCODER_EPOCHS) --learning-rate $(ENCODER_LEARNING_RATE) \
		--pos-weight-mode $(ENCODER_POS_WEIGHT_MODE)

train-direct: ## Prepare direct SFT splits and print the mlx_lm.lora command
	$(PY) scripts/train_direct_lora.py --data $(DATA)

train-marker: ## Prepare marker SFT splits and print the mlx_lm.lora command
	$(PY) scripts/train_marker_lora.py --data $(DATA)

train-lettucedetect: ## Train LettuceDetect feature classifier
	$(PY) scripts/train_lettucedetect.py --data $(DATA)

infer-direct: ## Inference with the trained direct adapter + evaluation
	$(PY) scripts/infer.py --data $(DATA) \
		--output results/direct_lora_predictions.jsonl \
		--mode direct --adapter-path results/adapters_direct
	$(PY) scripts/evaluate.py --data $(DATA) \
		--predictions results/direct_lora_predictions.jsonl \
		--output results/direct_lora_metrics.json

infer-marker: ## Inference with the trained marker adapter + evaluation
	$(PY) scripts/infer.py --data $(DATA) \
		--output results/marker_lora_predictions.jsonl \
		--mode marker --adapter-path results/adapters_marker
	$(PY) scripts/evaluate.py --data $(DATA) \
		--predictions results/marker_lora_predictions.jsonl \
		--output results/marker_lora_metrics.json

infer-lettucedetect: ## Inference with LettuceDetect classifier + evaluation
	$(PY) scripts/infer_lettucedetect.py --data $(DATA) \
		--model results/lettucedetect/classifier.joblib \
		--output results/lettucedetect/predictions.jsonl
	$(PY) scripts/evaluate.py --data $(DATA) \
		--predictions results/lettucedetect/predictions.jsonl \
		--output results/lettucedetect/metrics.json

serve-demo: ## Local manual web UI
	$(PY) scripts/serve_demo.py

benchmark-dummy: ## Unified benchmark smoke test with dummy methods
	$(PY) scripts/run_benchmark.py --data $(DATA) \
		--methods dummy_direct,dummy_marker \
		--output-dir results/benchmark_dummy

eval-all: ## Print every metrics json in results/
	@for f in results/*_metrics.json; do echo "== $$f"; cat "$$f"; done

clean: ## Remove tool caches and build artifacts (keeps results/)
	rm -rf .pytest_cache .ruff_cache .mypy_cache \
		src/*.egg-info src/rag_reliability/__pycache__ tests/__pycache__
