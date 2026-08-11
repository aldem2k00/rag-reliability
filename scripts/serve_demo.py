#!/usr/bin/env python
"""Serve a local manual demo UI for reliability judging methods."""

from __future__ import annotations

import argparse
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from rag_reliability.dataset import load_jsonl
from rag_reliability.dummy_model import DummyPredictor
from rag_reliability.methods import registry
from rag_reliability.methods.independent.predict import predict_independent
from rag_reliability.methods.lettucedetect.classifier import predictions_from_outputs
from rag_reliability.methods.lettucedetect.features import (
    FeatureConfig,
    extract_features,
    make_detector,
)
from rag_reliability.methods.m3 import build_user_prompt, parse_m3_prediction
from rag_reliability.mlx_backend import make_generate_fn
from rag_reliability.parsing import parse_prediction
from rag_reliability.prompts import build_direct_prompt, build_marker_prompt
from rag_reliability.schema import Prediction, RagSample


DEMO_CSS = """
.gradio-container {
    max-width: 1280px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-left: 24px !important;
    padding-right: 24px !important;
}
.method-note { color: #555; font-size: 0.92rem; }
"""

METHODS = registry.all_method_names()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--direct-adapter-path", default="results/adapters_direct")
    parser.add_argument("--marker-adapter-path", default="results/adapters_marker")
    parser.add_argument("--lettucedetect-model", default="results/lettucedetect/classifier.joblib")
    parser.add_argument("--encoder-model", default="deepvk/RuModernBERT-base")
    parser.add_argument("--encoder-checkpoint", default="results/encoder_checkpoints_512_best")
    parser.add_argument("--encoder-max-length", type=int, default=512)
    parser.add_argument("--encoder-threshold", type=float, default=0.72)
    parser.add_argument("--example-data", default="data/dummy.jsonl")
    return parser.parse_args()


def build_manual_sample(
    question: str,
    context: str,
    answer: str,
    faithfulness: int | None,
    relevance: int | None,
    marker: str | None,
) -> RagSample:
    return RagSample(
        id="manual_000001",
        question=question,
        context=context,
        answer=answer,
        faithfulness=1 if faithfulness is None else int(faithfulness),
        relevance=1 if relevance is None else int(relevance),
        marker=marker or "none",
    )


def prediction_payload(prediction: Prediction) -> dict[str, Any]:
    payload = prediction.model_dump()
    payload["reliable_pred"] = prediction.reliable_pred
    return payload


def gold_payload(sample: RagSample, has_gold: bool, prediction: Prediction) -> dict[str, Any] | None:
    if not has_gold:
        return None
    return {
        "faithfulness": sample.faithfulness,
        "relevance": sample.relevance,
        "reliable": sample.reliable,
        "faithfulness_correct": sample.faithfulness == prediction.faithfulness_pred,
        "relevance_correct": sample.relevance == prediction.relevance_pred,
        "reliable_correct": sample.reliable == prediction.reliable_pred,
    }


def method_statuses(
    direct_adapter_path: str = "results/adapters_direct",
    marker_adapter_path: str = "results/adapters_marker",
    lettucedetect_model: str = "results/lettucedetect/classifier.joblib",
    encoder_checkpoint: str = "results/encoder_checkpoints_512_best",
) -> dict[str, dict[str, Any]]:
    return {
        "dummy_direct": {"available": True, "artifact": None},
        "dummy_marker": {"available": True, "artifact": None},
        "prompt_direct": {"available": True, "artifact": "MLX model"},
        "prompt_marker": {"available": True, "artifact": "MLX model"},
        "lora_direct": {
            "available": Path(direct_adapter_path).exists(),
            "artifact": direct_adapter_path,
        },
        "lora_marker": {
            "available": Path(marker_adapter_path).exists(),
            "artifact": marker_adapter_path,
        },
        "lettucedetect": {
            "available": Path(lettucedetect_model).exists(),
            "artifact": lettucedetect_model,
        },
        "encoder": {
            "available": Path(encoder_checkpoint).exists(),
            "artifact": encoder_checkpoint,
        },
        "m3_zero_shot": {"available": True, "artifact": "MLX model"},
        "m3_few_shot": {
            "available": Path("configs/few_shot.yaml").exists(),
            "artifact": "configs/few_shot.yaml",
        },
        "m3_gepa": {
            "available": Path("configs/m3_gepa_prompt.txt").exists(),
            "artifact": "configs/m3_gepa_prompt.txt",
            "reason": "generate an evolved prompt with scripts/run_gepa.py",
        },
        "m3_openai": {
            "available": False,
            "artifact": None,
            "reason": "batch-only: requires OpenAI-compatible endpoint configuration",
        },
        "m3_openai_judge": {
            "available": False,
            "artifact": None,
            "reason": "batch-only: requires OpenAI-compatible endpoint configuration",
        },
        "m3_perchunk": {
            "available": False,
            "artifact": None,
            "reason": "batch-only: one request per chunk needs an OpenAI-compatible endpoint",
        },
        "ft_judge": {
            "available": False,
            "artifact": None,
            "reason": "batch-only: trains fold by fold via scripts/train_ft_judge.py",
        },
        "m6_selfcheck": {
            "available": False,
            "artifact": None,
            "reason": "batch-only: requires precomputed Method 6 feature JSONL",
        },
        "surface": {
            "available": False,
            "artifact": None,
            "reason": "batch-only: out-of-fold scoring needs the whole corpus and folds.json",
        },
        "majority": {
            "available": False,
            "artifact": None,
            "reason": "batch-only: out-of-fold scoring needs the whole corpus and folds.json",
        },
        "independent": {"available": True, "artifact": None},
        "independent_v2": {
            "available": Path("results/independent_v2/model.joblib").exists(),
            "artifact": "results/independent_v2/model.joblib",
        },
    }

def method_choice_labels(statuses: dict[str, dict[str, Any]]) -> list[str]:
    labels: list[str] = []
    for method, status in statuses.items():
        if status["available"]:
            state = "available"
        elif status.get("artifact"):
            state = f"missing: {status['artifact']}"
        else:
            state = status.get("reason", "unavailable")
        labels.append(f"{method} — {state}")
    return labels


def method_from_choice(choice: str) -> str:
    return choice.split(" — ", maxsplit=1)[0]


def normalize_methods(methods: list[str] | str | None) -> list[str]:
    if methods is None:
        return []
    raw_methods = [methods] if isinstance(methods, str) else methods
    return [method_from_choice(method) for method in raw_methods if method]


def example_choices(path: str, limit: int = 25) -> list[str]:
    try:
        samples = load_jsonl(path)
    except (FileNotFoundError, ValueError):
        return []
    return [
        f"{index} — {sample.id} — reliable={sample.reliable}"
        for index, sample in enumerate(samples[:limit])
    ]


def load_example_choice(choice: str | None, path: str) -> tuple[str, str, str, int | None, int | None, str]:
    if not choice:
        return "", "", "", None, None, "none"
    samples = load_jsonl(path)
    index = int(choice.split(" — ", maxsplit=1)[0])
    sample = samples[index]
    return (
        sample.question,
        sample.context,
        sample.answer,
        sample.faithfulness,
        sample.relevance,
        sample.marker or "none",
    )


@lru_cache(maxsize=8)
def cached_generate_fn(model: str, max_tokens: int, adapter_path: str | None):
    return make_generate_fn(model, max_tokens, adapter_path=adapter_path)


@lru_cache(maxsize=4)
def cached_lettucedetect_artifact(model_path: str):
    import joblib  # noqa: PLC0415

    artifact = joblib.load(model_path)
    saved_config = artifact.get("feature_config", {})
    config = FeatureConfig(
        model_path=saved_config.get("model_path") or FeatureConfig.model_path,
        threshold=saved_config.get("threshold", 0.5),
        device=saved_config.get("device"),
    )
    return artifact["pipeline"], config, make_detector(config)


def resolve_encoder_checkpoint(path: str) -> Path:
    checkpoint = Path(path)
    if checkpoint.is_file():
        return checkpoint
    if (checkpoint / "model.safetensors").exists() or (checkpoint / "pytorch_model.bin").exists():
        return checkpoint
    checkpoint_dirs = sorted(
        [child for child in checkpoint.glob("checkpoint-*") if child.is_dir()],
        key=lambda child: int(child.name.rsplit("-", maxsplit=1)[-1]),
    )
    return checkpoint_dirs[-1] if checkpoint_dirs else checkpoint


@lru_cache(maxsize=2)
def cached_encoder_artifact(
    encoder_model: str,
    encoder_checkpoint: str,
    max_length: int,
    threshold: float,
):
    import torch  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415

    from train_encoder_baseline import build_encoder_text, build_model  # noqa: PLC0415

    checkpoint = resolve_encoder_checkpoint(encoder_checkpoint)
    model = build_model(encoder_model, pos_weight=1.0)
    safetensors_path = checkpoint / "model.safetensors"
    pytorch_path = checkpoint / "pytorch_model.bin"
    if safetensors_path.exists():
        from safetensors.torch import load_file  # noqa: PLC0415

        state_dict = load_file(str(safetensors_path))
    elif pytorch_path.exists():
        state_dict = torch.load(pytorch_path, map_location="cpu")
    else:
        raise FileNotFoundError(f"Encoder checkpoint weights not found under {checkpoint}")
    model.load_state_dict(state_dict)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(encoder_model, trust_remote_code=True)
    return model, tokenizer, build_encoder_text, max_length, threshold


def run_prompt_method(
    sample: RagSample,
    mode: str,
    model: str,
    max_tokens: int,
    adapter_path: str | None = None,
) -> Prediction:
    build_prompt = build_direct_prompt if mode == "direct" else build_marker_prompt
    generate_fn = cached_generate_fn(model, max_tokens, adapter_path)
    raw_output = generate_fn(build_prompt(sample))
    return parse_prediction(raw_output, sample.id, expect_marker=(mode == "marker"))


def run_m3_method(sample: RagSample, model: str, max_tokens: int, mode: str = "zero_shot") -> Prediction:
    from rag_reliability.methods.m3 import build_system_prompt  # noqa: PLC0415

    generate_fn = cached_generate_fn(model, max_tokens, adapter_path=None)
    system_prompt = build_system_prompt(mode, examples_path="configs/few_shot.yaml")
    prompt = f"{system_prompt}\n\n{build_user_prompt(sample)}"
    raw_output = generate_fn(prompt)
    return parse_m3_prediction(raw_output, sample.id)


def run_lettucedetect_method(sample: RagSample, artifact_path: str) -> Prediction:
    pipeline, config, detector = cached_lettucedetect_artifact(artifact_path)
    features = extract_features([sample], detector, config.threshold, desc="features/manual")
    pred_y = pipeline.predict(features)
    return predictions_from_outputs([sample], np.asarray(pred_y), features)[0]


def run_encoder_method(
    sample: RagSample,
    encoder_model: str,
    encoder_checkpoint: str,
    max_length: int,
    threshold: float,
) -> Prediction:
    import torch  # noqa: PLC0415

    from train_encoder_baseline import predictions_from_probabilities  # noqa: PLC0415

    model, tokenizer, build_encoder_text, cached_max_length, cached_threshold = cached_encoder_artifact(
        encoder_model,
        encoder_checkpoint,
        max_length,
        threshold,
    )
    batch = tokenizer(
        build_encoder_text(sample),
        truncation=True,
        max_length=cached_max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])["logits"]
        probability = torch.sigmoid(logits).item()
    return predictions_from_probabilities([sample], [probability], threshold=cached_threshold)[0]


def run_manual_method(  # noqa: PLR0913, PLR0912
    method: str,
    question: str,
    context: str,
    answer: str,
    faithfulness: int | None = None,
    relevance: int | None = None,
    marker: str | None = None,
    model: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    max_tokens: int = 64,
    direct_adapter_path: str = "results/adapters_direct",
    marker_adapter_path: str = "results/adapters_marker",
    lettucedetect_model: str = "results/lettucedetect/classifier.joblib",
    encoder_model: str = "deepvk/RuModernBERT-base",
    encoder_checkpoint: str = "results/encoder_checkpoints_512_best",
    encoder_max_length: int = 512,
    encoder_threshold: float = 0.72,
) -> dict[str, Any]:
    statuses = method_statuses(
        direct_adapter_path,
        marker_adapter_path,
        lettucedetect_model,
        encoder_checkpoint,
    )
    if method not in statuses:
        return {"available": False, "error": f"Unknown method: {method}"}
    status = statuses[method]
    if not status["available"]:
        reason = status.get("reason") or f"Required artifact is missing: {status.get('artifact')}"
        return {"available": False, "method": method, "error": reason, "status": status}

    has_gold = faithfulness is not None and relevance is not None
    sample = build_manual_sample(question, context, answer, faithfulness, relevance, marker)

    try:
        if method.startswith("dummy_"):
            mode = method.removeprefix("dummy_")
            strategy = "keyword" if mode == "marker" else "always_reliable"
            raw_output = DummyPredictor(strategy=strategy, mode=mode).predict(sample)
            prediction = parse_prediction(raw_output, sample.id, expect_marker=(mode == "marker"))
        elif method.startswith("prompt_"):
            prediction = run_prompt_method(
                sample,
                mode=method.removeprefix("prompt_"),
                model=model,
                max_tokens=max_tokens,
            )
        elif method.startswith("lora_"):
            mode = method.removeprefix("lora_")
            prediction = run_prompt_method(
                sample,
                mode=mode,
                model=model,
                max_tokens=max_tokens,
                adapter_path=direct_adapter_path if mode == "direct" else marker_adapter_path,
            )
        elif method == "lettucedetect":
            prediction = run_lettucedetect_method(sample, lettucedetect_model)
        elif method == "encoder":
            prediction = run_encoder_method(
                sample,
                encoder_model=encoder_model,
                encoder_checkpoint=encoder_checkpoint,
                max_length=encoder_max_length,
                threshold=encoder_threshold,
            )
        elif method == "m3_zero_shot":
            prediction = run_m3_method(sample, model=model, max_tokens=max_tokens)
        elif method == "m3_few_shot":
            prediction = run_m3_method(
                sample,
                model=model,
                max_tokens=max_tokens,
                mode="few_shot",
            )
        elif method == "independent":
            prediction = predict_independent(sample)
        else:
            return {
                "available": False,
                "method": method,
                "error": status.get("reason", "Method is unavailable"),
                "status": status,
            }
    except Exception as exc:  # noqa: BLE001 - UI should surface method failures as data.
        return {"available": False, "method": method, "error": str(exc), "status": status}

    return {
        "available": True,
        "method": method,
        "prediction": prediction_payload(prediction),
        "gold": gold_payload(sample, has_gold, prediction),
    }


def result_row(result: dict[str, Any]) -> dict[str, Any]:
    if not result.get("available"):
        return {
            "method": result.get("method", "unknown"),
            "available": False,
            "reliable": None,
            "faithfulness": None,
            "relevance": None,
            "marker": None,
            "invalid": None,
            "correct": None,
            "error": result.get("error"),
        }

    prediction = result["prediction"]
    gold = result.get("gold")
    return {
        "method": result["method"],
        "available": True,
        "reliable": prediction["reliable_pred"],
        "faithfulness": prediction["faithfulness_pred"],
        "relevance": prediction["relevance_pred"],
        "marker": prediction.get("marker_pred"),
        "invalid": prediction["invalid_output"],
        "correct": None if gold is None else gold["reliable_correct"],
        "error": None,
    }


def readable_result(result: dict[str, Any]) -> str:
    if not result.get("available"):
        return f"{result.get('method', 'unknown')}: unavailable ({result.get('error')})"

    prediction = result["prediction"]
    reliable_text = "reliable" if prediction["reliable_pred"] else "unreliable"
    marker = prediction.get("marker_pred") or "none"
    gold = result.get("gold")
    correctness = ""
    if gold is not None:
        correctness = ", correct" if gold["reliable_correct"] else ", wrong"
    return (
        f"{result['method']}: reliable={prediction['reliable_pred']} ({reliable_text}), "
        f"faithfulness={prediction['faithfulness_pred']}, "
        f"relevance={prediction['relevance_pred']}, marker={marker}{correctness}"
    )


def raw_outputs(results: list[dict[str, Any]]) -> dict[str, Any]:
    rendered: dict[str, Any] = {}
    for result in results:
        method = result.get("method", "unknown")
        if result.get("available"):
            rendered[method] = result["prediction"].get("raw_output")
        else:
            rendered[method] = result.get("error")
    return rendered


def run_manual_methods(  # noqa: PLR0913
    methods: list[str] | str | None,
    question: str,
    context: str,
    answer: str,
    faithfulness: int | None = None,
    relevance: int | None = None,
    marker: str | None = None,
    model: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    max_tokens: int = 64,
    direct_adapter_path: str = "results/adapters_direct",
    marker_adapter_path: str = "results/adapters_marker",
    lettucedetect_model: str = "results/lettucedetect/classifier.joblib",
    encoder_model: str = "deepvk/RuModernBERT-base",
    encoder_checkpoint: str = "results/encoder_checkpoints_512_best",
    encoder_max_length: int = 512,
    encoder_threshold: float = 0.72,
) -> dict[str, Any]:
    normalized_methods = normalize_methods(methods)
    if not normalized_methods:
        return {
            "summary": "No methods selected.",
            "rows": [],
            "raw_outputs": {},
            "details": [],
        }

    results = [
        run_manual_method(
            method=method,
            question=question,
            context=context,
            answer=answer,
            faithfulness=faithfulness,
            relevance=relevance,
            marker=marker,
            model=model,
            max_tokens=max_tokens,
            direct_adapter_path=direct_adapter_path,
            marker_adapter_path=marker_adapter_path,
            lettucedetect_model=lettucedetect_model,
            encoder_model=encoder_model,
            encoder_checkpoint=encoder_checkpoint,
            encoder_max_length=encoder_max_length,
            encoder_threshold=encoder_threshold,
        )
        for method in normalized_methods
    ]
    return {
        "summary": "\n".join(readable_result(result) for result in results),
        "rows": [result_row(result) for result in results],
        "raw_outputs": raw_outputs(results),
        "details": results,
    }


def update_history(
    existing_history: list[dict[str, Any]] | None,
    new_rows: list[dict[str, Any]],
    max_rows: int = 25,
) -> list[dict[str, Any]]:
    history = list(existing_history or [])
    history.extend(new_rows)
    return history[-max_rows:]


def uploaded_file_path(uploaded_file: Any) -> str | None:
    if uploaded_file is None:
        return None
    if isinstance(uploaded_file, str):
        return uploaded_file
    if isinstance(uploaded_file, dict):
        path = uploaded_file.get("path") or uploaded_file.get("name")
        return str(path) if path else None
    name = getattr(uploaded_file, "name", None)
    return str(name) if name else None


def build_batch_command(
    data_path: str,
    methods: list[str] | str | None,
    output_dir: str,
    uploaded_file: Any = None,
) -> str:
    normalized_methods = normalize_methods(methods)
    selected_data = uploaded_file_path(uploaded_file) or data_path
    return (
        f"python scripts/run_benchmark.py --data {selected_data} "
        f"--methods {','.join(normalized_methods)} --output-dir {output_dir}"
    )


def run_ui_methods(  # noqa: PLR0913
    selected_methods: list[str] | str | None,
    question: str,
    context: str,
    answer: str,
    faithfulness: int | None,
    relevance: int | None,
    marker: str | None,
    model: str,
    max_tokens: int | float,
    direct_adapter_path: str,
    marker_adapter_path: str,
    lettucedetect_model: str,
    encoder_model: str,
    encoder_checkpoint: str,
    encoder_max_length: int | float,
    encoder_threshold: int | float,
    old_history: list[dict[str, Any]] | None,
) -> tuple[str, list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    result = run_manual_methods(
        methods=selected_methods,
        question=question,
        context=context,
        answer=answer,
        faithfulness=faithfulness,
        relevance=relevance,
        marker=marker,
        model=model,
        max_tokens=int(max_tokens),
        direct_adapter_path=direct_adapter_path,
        marker_adapter_path=marker_adapter_path,
        lettucedetect_model=lettucedetect_model,
        encoder_model=encoder_model,
        encoder_checkpoint=encoder_checkpoint,
        encoder_max_length=int(encoder_max_length),
        encoder_threshold=float(encoder_threshold),
    )
    new_history = update_history(old_history, result["rows"])
    return (
        result["summary"],
        result["rows"],
        result["raw_outputs"],
        result["details"],
        new_history,
        new_history,
    )


def load_first_example(path: str) -> RagSample | None:
    try:
        samples = load_jsonl(path)
    except (FileNotFoundError, ValueError):
        return None
    return samples[0] if samples else None


def build_ui(args: argparse.Namespace):
    try:
        import gradio as gr  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError('Install demo dependencies with: uv pip install -e ".[demo]"') from exc

    example = load_first_example(args.example_data)
    default_question = example.question if example is not None else ""
    default_context = example.context if example is not None else ""
    default_answer = example.answer if example is not None else ""
    default_faithfulness = example.faithfulness if example is not None else None
    default_relevance = example.relevance if example is not None else None
    default_marker = example.marker if example is not None else "none"

    statuses = method_statuses(
        args.direct_adapter_path,
        args.marker_adapter_path,
        args.lettucedetect_model,
        args.encoder_checkpoint,
    )
    method_labels = method_choice_labels(statuses)
    initial_examples = example_choices(args.example_data)

    with gr.Blocks(title="RAG Reliability Judge", css=DEMO_CSS) as demo:
        gr.Markdown("# RAG Reliability Judge")
        gr.Markdown(
            "<div class='method-note'>Manual single-example runner with dataset presets, "
            "multi-method comparison, raw outputs, history, and batch command generation.</div>"
        )
        with gr.Row():
            methods = gr.CheckboxGroup(
                choices=method_labels,
                value=[method_labels[0]],
                label="Methods",
            )
            faithfulness = gr.Radio(
                choices=[0, 1],
                value=default_faithfulness,
                label="Gold faithfulness",
            )
            relevance = gr.Radio(choices=[0, 1], value=default_relevance, label="Gold relevance")
        with gr.Row():
            example_data = gr.Textbox(args.example_data, label="Example dataset")
            example = gr.Dropdown(choices=initial_examples, label="Dataset preset")
            refresh_examples = gr.Button("Refresh presets")
            load_example = gr.Button("Load preset")

        with gr.Accordion("Method configuration", open=False):
            model = gr.Textbox(args.model, label="MLX model")
            max_tokens = gr.Number(args.max_tokens, label="Max generated tokens", precision=0)
            direct_adapter_path = gr.Textbox(args.direct_adapter_path, label="Direct adapter path")
            marker_adapter_path = gr.Textbox(args.marker_adapter_path, label="Marker adapter path")
            lettucedetect_model = gr.Textbox(args.lettucedetect_model, label="LettuceDetect artifact")
            encoder_model = gr.Textbox(args.encoder_model, label="Encoder base model")
            encoder_checkpoint = gr.Textbox(args.encoder_checkpoint, label="Encoder checkpoint")
            encoder_max_length = gr.Number(
                args.encoder_max_length,
                label="Encoder max length",
                precision=0,
            )
            encoder_threshold = gr.Number(args.encoder_threshold, label="Encoder threshold")

        gr.Markdown("## Input")
        question = gr.Textbox(default_question, label="Question / dialog", lines=6)
        context = gr.Textbox(default_context, label="Context", lines=10)
        answer = gr.Textbox(default_answer, label="Answer", lines=5)
        marker = gr.Textbox(default_marker or "none", label="Gold marker")
        run_button = gr.Button("Run method", variant="primary")
        gr.Markdown("## Results")
        summary = gr.Textbox(label="Summary", lines=6)
        table = gr.Dataframe(label="Comparison", interactive=False)
        raw_output = gr.JSON(label="Raw outputs")
        details = gr.JSON(label="Details")
        history_state = gr.State([])
        history = gr.Dataframe(label="History", interactive=False)
        gr.JSON(
            label="Method availability",
            value=statuses,
        )

        with gr.Accordion("Batch command", open=False):
            batch_data = gr.Textbox(args.example_data, label="Batch data path")
            batch_upload = gr.File(label="Optional JSONL upload", file_count="single")
            batch_output_dir = gr.Textbox("results/demo_batch", label="Batch output dir")
            batch_button = gr.Button("Build batch command")
            batch_command = gr.Textbox(label="Command")

        refresh_examples.click(
            fn=lambda path: gr.update(choices=example_choices(path)),
            inputs=example_data,
            outputs=example,
        )
        load_example.click(
            fn=load_example_choice,
            inputs=[example, example_data],
            outputs=[question, context, answer, faithfulness, relevance, marker],
        )
        run_button.click(
            fn=run_ui_methods,
            inputs=[
                methods,
                question,
                context,
                answer,
                faithfulness,
                relevance,
                marker,
                model,
                max_tokens,
                direct_adapter_path,
                marker_adapter_path,
                lettucedetect_model,
                encoder_model,
                encoder_checkpoint,
                encoder_max_length,
                encoder_threshold,
                history_state,
            ],
            outputs=[summary, table, raw_output, details, history_state, history],
        )
        batch_button.click(
            fn=build_batch_command,
            inputs=[batch_data, methods, batch_output_dir, batch_upload],
            outputs=batch_command,
        )

    return demo


def main() -> None:
    args = parse_args()
    demo = build_ui(args)
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()