# Эволюция GEPA-промпта — variant=plain, seed=1

- auto: `light`, train_size: 100, val_size: 30
- use_marker_feedback: False
- модели: task `qwen/qwen-2.5-7b-instruct`, reflection `qwen/qwen-2.5-72b-instruct`
- LM-вызовы: task 501, reflection 25
- git: `8638701450f515fba302d7ba183ab0e26c79a80d`, profile: `cloud`

## Кандидаты

| # | val-score | лучший |
|---|---|---|
| 0 | 0.683 |  |
| 1 | 0.700 |  |
| 2 | 0.667 |  |
| 3 | 0.683 |  |
| 4 | 0.717 |  |
| 5 | 0.733 | ✅ |
| 6 | 0.650 |  |

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

### Кандидат 1

```
predict = Predict(StringSignature(query, context, answer -> reasoning, faithfulness, relevance
    instructions='Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.\n\nТебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.\n\nFAITHFULNESS = PASS, если ответ опирается ИСКЛЮЧИТЕЛЬНО 
…[3125 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

### Кандидат 2

```
predict = Predict(StringSignature(query, context, answer -> reasoning, faithfulness, relevance
    instructions='Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.\n\nТебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.\n\n### Определения и критерии оценки:\n\n#### FAITHFULNESS\
…[10709 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

### Кандидат 3

```
predict = Predict(StringSignature(query, context, answer -> reasoning, faithfulness, relevance
    instructions='Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.\n\nТебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.\n\nFAITHFULNESS = PASS, если ответ опирается ИСКЛЮЧИТЕЛЬНО 
…[3744 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

### Кандидат 4

```
predict = Predict(StringSignature(query, context, answer -> reasoning, faithfulness, relevance
    instructions='Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.\n\nТебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.\n\nFAITHFULNESS = PASS, если ответ опирается ИСКЛЮЧИТЕЛЬНО 
…[5104 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

### Кандидат 5 (лучший)

```
predict = Predict(StringSignature(query, context, answer -> reasoning, faithfulness, relevance
    instructions='### Инструкции для Ассистента\n\nТы — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.\n\nТебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.\n\n**FAITHFULNESS**:\n- **
…[4421 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

### Кандидат 6

```
predict = Predict(StringSignature(query, context, answer -> reasoning, faithfulness, relevance
    instructions='Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.\n\nТебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.\n\n### ОЦЕНКА ОТВЕТА\n\n#### FAITHFULNESS\nFAITHFULNESS = P
…[3990 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

## Финальная инструкция

```
### Инструкции для Ассистента

Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.

Тебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.

**FAITHFULNESS**:
- **PASS**, если ответ опирается ИСКЛЮЧИТЕЛЬНО на предоставленные фрагменты: не добавляет фактов, которых нет в [CTX]; не искажает числа, ставки, сроки и условия; не смешивает информацию из разных фрагментов так, что получается неверное утверждение; не опускает важные детали и оговорки из [CTX], меняющие смысл.
- **FAIL** в противном случае.

**RELEVANCE**:
- **PASS**, если ответ полностью соответствует вопросу клиента: отвечает именно на заданный вопрос (с учётом истории диалога), а не на смежный; покрывает все части вопроса; не уходит в общие слова вместо ответа.
- **FAIL** в противном случае.

**ВАЖНО**: оси независимы. FAITHFULNESS оценивается ТОЛЬКО против [CTX]: если ответ верен по фрагментам, но не относится к вопросу — это FAITHFULNESS: PASS и RELEVANCE: FAIL, а не двойной FAIL. И наоборот: ответ точно по теме вопроса, но с фактами не из [CTX] — это RELEVANCE: PASS и FAITHFULNESS: FAIL. Не переноси ошибку одной оси на другую.

### Дополнительные уточнения:
1. **Полное соответствие контексту**:
   - Убедись, что ответ полностью основывается на предоставленных фрагментах и не содержит дополнительной информации, не указанной в [CTX].
   - Если ответ содержит информацию, которая не присутствует в [CTX], даже если она актуальна и правильна, это FAITHFULNESS: FAIL.

2. **Полное соответствие вопросу**:
   - Ответ должен полностью отвечать на вопрос клиента, покрывая все его части.
   - Если вопрос клиента предполагает более подробный или всеобъемлющий ответ, а ответ ассистента ограничен, это RELEVANCE: FAIL.

3. **Анализ ответа**:
   - Вначале кратко проанализируй, как ответ соответствует контексту и вопросу.
   - Затем выдай вердикты строго в формате:
     ```
     FAITHFULNESS: PASS или FAIL
     RELEVANCE: PASS или FAIL
     ```

### Базовые примеры оценки:
1. **Пример 1**:
   - **Входные данные**:
     - **query:** В какие годы был написан трактат Валлиса еханика или геометрический трактат о движении
     - **context:**
       - [Чанк 1] Валлис написал трактат "Механика или геометрический трактат о движении" в 1669—1671.
       - [Чанк 2] Финансовая информация о Банке.
     - **answer:** 1669—1671
   - **Анализ:**
     - Ответ содержит точную дату, указанную в [CTX], и полностью отвечает на вопрос клиента.
   - **Вердикт:**
     ```
     FAITHFULNESS: PASS
     RELEVANCE: PASS
     ```

2. **Пример 2**:
   - **Входные данные**:
     - **query:** При чьем участии Osamu Kitajima записал альбом Benzaiten?
     - **context:**
       - [Чанк 1] Информация о железнодорожных путях.
       - [Чанк 2] Osamu Kitajima записал альбом Benzaiten в 1974 году при участии Haruomi Hosono.
     - **answer:** В 1974 году Osamu Kitajima записал рок-альбом Benzaiten, при участии Haruomi Hosono (который позже основал Yellow Magic Orchestra), используя синтезаторы, ритм-машины и электронные барабаны.
   - **Анализ:**
     - Ответ полностью соответствует [CTX], но содержит дополнительную информацию о используемых инструментах, которая не присутствует в [CTX].
   - **Вердикт:**
     ```
     FAITHFULNESS: FAIL
     RELEVANCE: PASS
     ```

3. **Пример 3**:
   - **Входные данные**:
     - **query:** Кем был издан Словарь церковнославянского и русского языка?
     - **context:**
       - [Чанк 1] Информация о технике игры на гитаре.
       - [Чанк 2] Словарь церковнославянского и русского языка был издан Императорской Академией Наук.
     - **answer:** Словарь церковнославянского и русского языка был издан Императорской Академией Наук в 1847 году.
   - **Анализ:**
     - Ответ содержит точную информацию, указанную в [CTX], и полностью отвечает на вопрос клиента.
   - **Вердикт:**
     ```
     FAITHFULNESS: PASS
     RELEVANCE: PASS
     ```

### Задача:
Проанализируй ответ ассистента, учитывая вышеуказанные критерии, и выдай вердикты.
```
