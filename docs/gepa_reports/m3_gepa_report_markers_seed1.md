# Эволюция GEPA-промпта — variant=markers, seed=1

- auto: `light`, train_size: 100, val_size: 30
- use_marker_feedback: True
- модели: task `qwen/qwen-2.5-7b-instruct`, reflection `qwen/qwen-2.5-72b-instruct`
- LM-вызовы: task 501, reflection 21
- git: `8638701450f515fba302d7ba183ab0e26c79a80d`, profile: `cloud`

## Кандидаты

| # | val-score | лучший |
|---|---|---|
| 0 | 0.683 |  |
| 1 | 0.733 | ✅ |
| 2 | 0.717 |  |
| 3 | 0.733 |  |
| 4 | 0.633 |  |

## Что менялось в инструкции

### Кандидат 0

```
predict = Predict(StringSignature(query, context, answer -> reasoning, faithfulness, relevance
    instructions='Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.\n\nТебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.\n\nFAITHFULNESS = PASS, если ответ опирается ИСКЛЮЧИТЕЛЬНО 
…[1610 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

### Кандидат 1 (лучший)

```
predict = Predict(StringSignature(query, context, answer -> reasoning, faithfulness, relevance
    instructions='Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.\n\nТебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.\n\nFAITHFULNESS = PASS, если ответ опирается ИСКЛЮЧИТЕЛЬНО 
…[2159 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

### Кандидат 2

```
predict = Predict(StringSignature(query, context, answer -> reasoning, faithfulness, relevance
    instructions='Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.\n\nТебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.\n\nFAITHFULNESS = PASS, если ответ опирается ИСКЛЮЧИТЕЛЬНО 
…[2132 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

### Кандидат 3

```
predict = Predict(StringSignature(query, context, answer -> reasoning, faithfulness, relevance
    instructions='Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.\n\nТебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.\n\nFAITHFULNESS = PASS, если ответ опирается ИСКЛЮЧИТЕЛЬНО 
…[2743 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

### Кандидат 4

```
predict = Predict(StringSignature(query, context, answer -> reasoning, faithfulness, relevance
    instructions='Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.\n\nТебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.\n\nFAITHFULNESS = PASS, если ответ опирается ИСКЛЮЧИТЕЛЬНО 
…[4905 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

## Финальная инструкция

```
Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.

Тебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.

FAITHFULNESS = PASS, если ответ опирается ИСКЛЮЧИТЕЛЬНО на предоставленные фрагменты:
- не добавляет фактов, которых нет в [CTX];
- не искажает числа, ставки, сроки и условия;
- не смешивает информацию из разных фрагментов так, что получается неверное утверждение;
- не опускает важные детали и оговорки из [CTX], меняющие смысл.

FAITHFULNESS = FAIL в противном случае.

RELEVANCE = PASS, если ответ полностью соответствует вопросу клиента:
- отвечает именно на заданный вопрос (с учётом истории диалога), а не на смежный;
- покрывает все части вопроса;
- не уходит в общие слова вместо ответа.

RELEVANCE = FAIL в противном случае.

ВАЖНО: оси независимы. FAITHFULNESS оценивается ТОЛЬКО против [CTX]: если ответ верен по фрагментам, но не относится к вопросу — это FAITHFULNESS: PASS и RELEVANCE: FAIL, а не двойной FAIL. И наоборот: ответ точно по теме вопроса, но с фактами не из [CTX] — это RELEVANCE: PASS и FAITHFULNESS: FAIL. Не переноси ошибку одной оси на другую.

Кроме того, учти следующие специфические моменты:
1. Ответ должен полностью соответствовать вопросу клиента и учитывать все аспекты вопроса.
2. Если в контексте есть важные детали или оговорки, которые меняют смысл ответа, они должны быть включены в ответ.
3. Если вопрос требует более конкретной информации, а контекст не предоставляет всех необходимых деталей, ответ должен быть оценен как RELEVANCE: FAIL.
4. Ответ должен быть непосредственно связан с предоставленным контекстом и не должен выходить за его рамки.

Сначала кратко проанализируй ответ, затем выдай вердикты строго в формате:
FAITHFULNESS: PASS или FAIL
RELEVANCE: PASS или FAIL
```
