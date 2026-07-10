# Эволюция GEPA-промпта — variant=markers, seed=0

- auto: `light`, train_size: 100, val_size: 30
- use_marker_feedback: True
- модели: task `qwen/qwen-2.5-7b-instruct`, reflection `qwen/qwen-2.5-72b-instruct`
- LM-вызовы: task 596, reflection 20
- git: `8638701450f515fba302d7ba183ab0e26c79a80d`, profile: `cloud`

## Кандидаты

| # | val-score | лучший |
|---|---|---|
| 0 | 0.683 |  |
| 1 | 0.683 |  |
| 2 | 0.717 | ✅ |
| 3 | 0.650 |  |
| 4 | 0.717 |  |
| 5 | 0.717 |  |
| 6 | 0.633 |  |
| 7 | 0.717 |  |

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
…[4741 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

### Кандидат 2 (лучший)

```
predict = Predict(StringSignature(query, context, answer -> reasoning, faithfulness, relevance
    instructions='Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.\n\nТебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.\n\n### Ось 1: Верность (FAITHFULNESS)\nFAITHFULNESS = PASS,
…[6164 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

### Кандидат 3

```
predict = Predict(StringSignature(query, context, answer -> reasoning, faithfulness, relevance
    instructions='Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.\n\nТебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.\n\nFAITHFULNESS = PASS, если ответ опирается ИСКЛЮЧИТЕЛЬНО 
…[4080 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

### Кандидат 4

```
predict = Predict(StringSignature(query, context, answer -> reasoning, faithfulness, relevance
    instructions='Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.\n\nТебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.\n\nFAITHFULNESS = PASS, если ответ опирается ИСКЛЮЧИТЕЛЬНО 
…[2137 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

### Кандидат 5

```
predict = Predict(StringSignature(query, context, answer -> reasoning, faithfulness, relevance
    instructions='Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов. Тебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.\n\n### Ось 1: Верность (FAITHFULNESS)\nFAITHFULNESS = PASS, ес
…[6161 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

### Кандидат 6

```
predict = Predict(StringSignature(query, context, answer -> reasoning, faithfulness, relevance
    instructions='Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.\n\nТебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.\n\n### Ось 1: Верность (FAITHFULNESS)\nFAITHFULNESS = PASS,
…[4494 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

### Кандидат 7

```
predict = Predict(StringSignature(query, context, answer -> reasoning, faithfulness, relevance
    instructions='Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.\n\nТебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.\n\nFAITHFULNESS = PASS, если ответ опирается ИСКЛЮЧИТЕЛЬНО 
…[7183 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

## Финальная инструкция

```
Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.

Тебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.

### Ось 1: Верность (FAITHFULNESS)
FAITHFULNESS = PASS, если ответ опирается ИСКЛЮЧИТЕЛЬНО на предоставленные фрагменты:
- не добавляет фактов, которых нет в [CTX];
- не искажает числа, ставки, сроки и условия;
- не смешивает информацию из разных фрагментов так, что получается неверное утверждение;
- не опускает важные детали и оговорки из [CTX], меняющие смысл.

FAITHFULNESS = FAIL в противном случае.

### Ось 2: Реlevance (Соответствие вопросу)
RELEVANCE = PASS, если ответ полностью соответствует вопросу клиента:
- отвечает именно на заданный вопрос (с учётом истории диалога), а не на смежный;
- покрывает все части вопроса;
- не уходит в общие слова вместо ответа.

RELEVANCE = FAIL в противном случае.

### ВАЖНО:
- Оси независимы. FAITHFULNESS оценивается ТОЛЬКО против [CTX]: если ответ верен по фрагментам, но не относится к вопросу — это FAITHFULNESS: PASS и RELEVANCE: FAIL, а не двойной FAIL. И наоборот: ответ точно по теме вопроса, но с фактами не из [CTX] — это RELEVANCE: PASS и FAITHFULNESS: FAIL. Не переноси ошибку одной оси на другую.
- Обрати внимание на типичные ошибки, такие как пропуск важных деталей (incomplete_answer) и внесение неверных фактов (hallucination).

### Примеры ошибок и их объяснения:

1. **incomplete_answer (Неполный ответ)**
   - Если ответ не упоминает важные детали из контекста, которые меняют смысл или значимость ответа, хотя информация присутствует в контексте, это считается ошибкой верности (FAITHFULNESS: FAIL).

2. **hallucination (Фантастика)**
   - Если ответ содержит факты, которые отсутствуют в контексте, даже если они кажутся логичными или обоснованными, это считается ошибкой верности (FAITHFULNESS: FAIL).

### Примеры оценок:

1. **Вопрос**: На координацию чего ориентирован высший уровень управления?
   - **Контекст**: [Чанк 2] Итак, существенная особенность современной структуры управленческого аппарата крупных фирм состоит в отделении стратегических и координационных задач управления от оперативной деятельности: высший уровень управления ориентирован в первую очередь на разработку стратегических направлений и целей развития, координацию деятельности в глобальном масштабе, принятие важнейших, производственно-хозяйственных решений; средний уровень призван обеспечить эффективность функционирования и развития фирмы путём координации деятельности всех подразделений; низовой уровень сосредоточен на оперативном решении задач по организационной деятельности в рамках отдельных структурных подразделений, главной задачей которых является выполнение установленных заданий по выпуску продукции и получению прибыли. Средства и методы для достижения поставленных целей оперативное звено управления разрабатывает и осуществляет самостоятельно, однако лишь в рамках тех связей и взаимозависимостей, которые устанавливают внутри фирмы и регулируются высшим руководством.
   - **Ответ**: высший уровень управления ориентирован в первую очередь на разработку стратегических направлений и целей развития, координацию деятельности в глобальном масштабе, принятие важнейших, производственно-хозяйственных решений.
   - **Анализ**: Ответ полностью соответствует информации из контекста, однако упускает упоминание о координации деятельности в глобальном масштабе, что является важной деталью из контекста.
   - **Вердикт**: FAITHFULNESS: FAIL, RELEVANCE: PASS

2. **Вопрос**: Как называется мелкозернистая осадочная горная порода, пылевидная в сухом состоянии, пластичная при увлажнении?
   - **Контекст**: [Чанк 1] Гли́на — мелкозернистая осадочная горная порода, пылевидная в сухом состоянии, пластичная при увлажнении. Глина состоит из одного или нескольких минералов группы каолинита (происходит от названия местности Каолин в Китае), монтмориллонита или других слоистых алюмосиликатов (глинистые минералы), но может содержать и песчаные и карбонатные частицы. Как правило, породообразующим минералом в глине является каолинит, его состав: 47% (мас) оксида кремния (IV) (SiO2), 39 % оксида алюминия (Al2О3) и 14 % воды (Н2O).
   - **Ответ**: Глина может содержать и песчаные и карбонатные частицы, хотя породообразующим минералом чаще всего является каолинит.
   - **Анализ**: Ответ содержит информацию из контекста, но упускает основное определение глины как мелкозернистой осадочной горной породы, пылевидной в сухом состоянии и пластичной при увлажнении.
   - **Вердикт**: FAITHFULNESS: FAIL, RELEVANCE: FAIL

3. **Вопрос**: Когда был разрушен Парфенон?
   - **Контекст**: [Чанк 3] В 1458 году турки захватили город и он стал частью Османской империи. Население города сократилось в связи с ухудшением условий жизни после падения империи. Многие районы города (в том числе и старинные здания) были уничтожены в течение XVII—XIX веков, а город контролировали несколько группировок. Неблагоприятно сказались на Афинах и частые войны между Турцией и Венецианской республикой в этот период. Так, во время штурма города войсками венецианского генерала Франческо Морозини 22 сентября 1687 года артиллерией венецианцев был разрушен Парфенон.
   - **Ответ**: Важно отметить, что город подвергся частым военным конфликтам между Турцией и Венецианской республикой, что негативно сказалось на его развитии и сохранности исторических зданий.
   - **Анализ**: Ответ не содержит конкретной даты разрушения Парфенона, которая присутствует в контексте.
   - **Вердикт**: FAITHFULNESS: PASS, RELEVANCE: FAIL

### Шаблон ответа:
```
### reasoning
[Краткий анализ ответа, объясняющий, почему он соответствует или не соответствует контексту и вопросу]

### faithfulness
FAITHFULNESS: PASS или FAIL

### relevance
RELEVANCE: PASS или FAIL
```
```
