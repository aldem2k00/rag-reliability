"""Method 3 prompts ported from the m3-m6 branch.

The prompt keeps the original PASS/FAIL contract, while runners convert that
contract into the repository-wide ``Prediction`` JSONL format.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from rag_reliability.schema import RagSample

SEED_INSTRUCTION = """Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.

Тебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.

FAITHFULNESS = PASS, если ответ опирается ИСКЛЮЧИТЕЛЬНО на предоставленные фрагменты: не добавляет фактов, которых нет в [CTX]; не искажает числа, ставки, сроки и условия; не смешивает информацию из разных фрагментов так, что получается неверное утверждение; не опускает важные детали и оговорки из [CTX], меняющие смысл.
FAITHFULNESS = FAIL в противном случае.

RELEVANCE = PASS, если ответ полностью соответствует вопросу клиента: отвечает именно на заданный вопрос (с учётом истории диалога), а не на смежный; покрывает все части вопроса; не уходит в общие слова вместо ответа.
RELEVANCE = FAIL в противном случае.

ВАЖНО: оси независимы. FAITHFULNESS оценивается ТОЛЬКО против [CTX]: если ответ верен по фрагментам, но не относится к вопросу — это FAITHFULNESS: PASS и RELEVANCE: FAIL, а не двойной FAIL. И наоборот: ответ точно по теме вопроса, но с фактами не из [CTX] — это RELEVANCE: PASS и FAITHFULNESS: FAIL. Не переноси ошибку одной оси на другую.

Сначала кратко проанализируй ответ, затем выдай вердикты строго в формате:
FAITHFULNESS: PASS или FAIL
RELEVANCE: PASS или FAIL"""

USER_TEMPLATE = """[Q]
{question}

[CTX]
{context}

[A]
{answer}

Проанализируй и выдай вердикты в заданном формате."""

FEW_SHOT_TEMPLATE = """

Пример {index}.
[Q] {q}
[CTX] {ctx}
[A] {a}
Анализ: {analysis}
FAITHFULNESS: {faith}
RELEVANCE: {rel}"""


def build_few_shot_system(examples: list[dict]) -> str:
    blocks = [
        FEW_SHOT_TEMPLATE.format(index=index + 1, **example)
        for index, example in enumerate(examples)
    ]
    return SEED_INSTRUCTION + "".join(blocks)


def build_system_prompt(
    mode: str,
    *,
    examples_path: str | Path | None = None,
    prompt_file: str | Path | None = None,
) -> str:
    if mode == "zero_shot":
        return SEED_INSTRUCTION
    if mode == "few_shot":
        if examples_path is None:
            raise ValueError("few_shot mode requires examples_path")
        payload = yaml.safe_load(Path(examples_path).read_text(encoding="utf-8"))
        return build_few_shot_system(payload["examples"])
    if mode == "gepa":
        if prompt_file is None:
            raise ValueError("gepa mode requires prompt_file")
        path = Path(prompt_file)
        if not path.exists():
            raise FileNotFoundError(
                f"GEPA prompt file not found: {path}. Generate one with scripts/run_gepa.py "
                "(results/gepa/m3_optimized_prompt_<variant>_seed<seed>.txt) or use the "
                "committed configs/m3_gepa_prompt.txt"
            )
        return path.read_text(encoding="utf-8").strip()
    raise ValueError(f"Unknown Method 3 mode: {mode}")


def build_user_prompt(sample: RagSample, max_context_chars: int | None = None) -> str:
    context = sample.context
    if max_context_chars is not None and len(context) > max_context_chars:
        context = context[:max_context_chars] + "\n[контекст усечён]"
    return USER_TEMPLATE.format(
        question=sample.question,
        context=context,
        answer=sample.answer,
    )
