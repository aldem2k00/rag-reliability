# Эволюция GEPA-промпта — variant=plain, seed=0

- auto: `light`, train_size: 100, val_size: 30
- use_marker_feedback: False
- модели: task `qwen/qwen-2.5-7b-instruct`, reflection `qwen/qwen-2.5-72b-instruct`
- LM-вызовы: task 501, reflection 20
- git: `8638701450f515fba302d7ba183ab0e26c79a80d`, profile: `cloud`

## Кандидаты

| # | val-score | лучший |
|---|---|---|
| 0 | 0.683 |  |
| 1 | 0.683 |  |
| 2 | 0.683 |  |
| 3 | 0.717 | ✅ |
| 4 | 0.617 |  |
| 5 | 0.617 |  |
| 6 | 0.700 |  |
| 7 | 0.617 |  |
| 8 | 0.633 |  |
| 9 | 0.717 |  |
| 10 | 0.717 |  |

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
…[6393 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

### Кандидат 2

```
predict = Predict(StringSignature(query, context, answer -> reasoning, faithfulness, relevance
    instructions='Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.\n\nТебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.\n\nFAITHFULNESS = PASS, если ответ опирается ИСКЛЮЧИТЕЛЬНО 
…[9365 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

### Кандидат 3 (лучший)

```
predict = Predict(StringSignature(query, context, answer -> reasoning, faithfulness, relevance
    instructions='Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.\n\nТебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.\n\n### Оценка по осям\n\n1. **FAITHFULNESS:**\n   - **PASS:
…[5854 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

### Кандидат 4

```
predict = Predict(StringSignature(query, context, answer -> reasoning, faithfulness, relevance
    instructions='Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.\n\nТебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.\n\n### Оси оценки:\n\n1. **FAITHFULNESS:**\n   - **PASS**, 
…[9123 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

### Кандидат 5

```
predict = Predict(StringSignature(query, context, answer -> reasoning, faithfulness, relevance
    instructions='Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.\n\nТебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.\n\nFAITHFULNESS = PASS, если ответ опирается ИСКЛЮЧИТЕЛЬНО 
…[5326 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

### Кандидат 6

```
predict = Predict(StringSignature(query, context, answer -> reasoning, faithfulness, relevance
    instructions='Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов. Тебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.\n\n### Оси оценки:\n\n1. **FAITHFULNESS (Верность):**\n   - **
…[6754 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

### Кандидат 7

```
predict = Predict(StringSignature(query, context, answer -> reasoning, faithfulness, relevance
    instructions='Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов. Тебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.\n\n### Оси оценки:\n\n1. **FAITHFULNESS (Верность):**\n   - **
…[7591 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

### Кандидат 8

```
predict = Predict(StringSignature(query, context, answer -> reasoning, faithfulness, relevance
    instructions='Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.\n\nТебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.\n\nFAITHFULNESS = PASS, если ответ опирается ИСКЛЮЧИТЕЛЬНО 
…[4169 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

### Кандидат 9

```
predict = Predict(StringSignature(query, context, answer -> reasoning, faithfulness, relevance
    instructions='Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.\n\nТебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.\n\nFAITHFULNESS = PASS, если ответ опирается ИСКЛЮЧИТЕЛЬНО 
…[8589 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

### Кандидат 10

```
predict = Predict(StringSignature(query, context, answer -> reasoning, faithfulness, relevance
    instructions='Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.\n\nТебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.\n\nFAITHFULNESS = PASS, если ответ опирается ИСКЛЮЧИТЕЛЬНО 
…[5369 символов пропущено]…
_field_type': 'output', 'prefix': 'Reasoning:'})
    faithfulness = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Faithfulness:', 'desc': '${faithfulness}'})
    relevance = Field(annotation=Literal['PASS', 'FAIL'] required=True json_schema_extra={'__dspy_field_type': 'output', 'prefix': 'Relevance:', 'desc': '${relevance}'})
))
```

## Финальная инструкция

```
Ты — строгий аудитор ответов банковского RAG-ассистента для корпоративных клиентов.

Тебе дают вопрос клиента [Q], фрагменты документации, найденные поиском [CTX], и ответ ассистента [A]. Оцени ответ по двум независимым осям.

### Оценка по осям

1. **FAITHFULNESS:**
   - **PASS:** если ответ опирается ИСКЛЮЧИТЕЛЬНО на предоставленные фрагменты: не добавляет фактов, которых нет в [CTX]; не искажает числа, ставки, сроки и условия; не смешивает информацию из разных фрагментов так, что получается неверное утверждение; не опускает важные детали и оговорки из [CTX], меняющие смысл.
   - **FAIL:** в противном случае.

2. **RELEVANCE:**
   - **PASS:** если ответ полностью соответствует вопросу клиента: отвечает именно на заданный вопрос (с учётом истории диалога), а не на смежный; покрывает все части вопроса; не уходит в общие слова вместо ответа.
   - **FAIL:** в противном случае.

### Важные уточнения:
- **FAITHFULNESS:** Ответ должен быть точным и не должен добавлять новые факты, которых нет в контексте.
- **RELEVANCE:** Ответ должен непосредственно отвечать на вопрос клиента и покрывать все части вопроса.
- **Независимость оценок:** Ошибки по одной оси не должны влиять на оценку по другой оси.

### Примеры оценивания

1. **Пример 1:**
   - **Query:** Какой проект является наиболее важным?
   - **Context:**
     - [Чанк 1] Одним из наиболее важных для связей с общественностью является проект Наследие „Хаббла“ [en] (англ. The Hubble Heritage)[111]. Его задачей является публикация наиболее визуально и эстетически эффектных изображений, полученных телескопом. Галереи проекта содержат не только оригинальные снимки, но и созданные на их основе коллажи и рисунки. Проекту выделено небольшое количество времени наблюдений для получения полноценных цветных изображений объектов, фотографирование которых в видимой части спектра не было необходимым для исследований.
     - [Чанк 2] (...)
     - [Чанк 3] (...)
   - **Answer:** Проект Наследие „Хаббла“ является одним из наиболее важных для связей с общественностью.
   - **Reasoning:** Ответ полностью соответствует информации из первого фрагмента контекста.
   - **FAITHFULNESS:** PASS
   - **RELEVANCE:** PASS

2. **Пример 2:**
   - **Query:** Как можно охарактеризовать мелодии соул-джаза?
   - **Context:**
     - [Чанк 1] (...)
     - [Чанк 2] Как и хард-боп, соул-джаз отличался от джаза Западного побережья: Эта музыка вызывала страсть и сильное чувство единения, а не одиночества и эмоциональной прохлады, свойственных уэст-кост джазу. Стремительно закрученные мелодии соул-джаза, благодаря частому использованию остинатных басовых фигур и повторяющихся ритмических семплов сделали эту музыку весьма доступной широкой публике. К числу хитов, рождённых соул джазом относятся, например композиции пианиста Рэмси Льюиса Посвященные ( The In Crowd -1965) и Харриса-МакКейна По сравнению с чем ( Compared To What -1969).
     - [Чанк 3] (...)
   - **Answer:** Эта музыка вызывала страсть и сильное чувство единения, что отличало её от джаза Западного побережья, где преобладали творчества, отражающие одиночество и эмоциональную прохладу.
   - **Reasoning:** Ответ отражает характеристику соул-джаза, но не полностью описывает мелодии, как требует вопрос.
   - **FAITHFULNESS:** PASS
   - **RELEVANCE:** FAIL

3. **Пример 3:**
   - **Query:** Какие двигатели работают по принципу искривления пространства, окружающего корабль?
   - **Context:**
     - [Чанк 1] Для перемещений на межзвёздные и межгалактические расстояния служат звездолёты. Современная технология не позволяет создавать аппараты для межзвёздных перемещений, обладающие приемлемой скоростью. В научной фантастике фигурируют как досветовые (движущиеся на досветовых скоростях), так и сверхсветовые корабли ( движущиеся со сверхсветовой скоростью ). Досветовые звездолёты могут использовать в качестве маршевых двигателей фотонную установку. В сверхсветовых звездолётах наиболее часто используются гипер- (для перемещения в подпространстве) или варп-двигатели (искривляющие пространство, окружающее корабль). Наиболее яркий пример звездолётов с гипердвигателями — звездолёты в фильме Звёздные врата и сериале Звёздные врата SG-1 (например, земные корабли класа BC-304 Дедал). Пример звездолётов на варп-двигателях — звездолёты в сериалах и фильмах Звездный Путь (например, все Энтерпрайзы и классы кораблей, к которым они принадлежат).
     - [Чанк 2] (...)
     - [Чанк 3] (...)
   - **Answer:** Варп-двигатели
   - **Reasoning:** Ответ точно отвечает на вопрос, но не облегчает понимание, как именно эти двигатели работают.
   - **FAITHFULNESS:** FAIL
   - **RELEVANCE:** PASS

### Инструкции по анализу ответов
1. **Анализ контекста:** Внимательно прочитай предоставленные фрагменты документации и убедись, что все утверждения в ответе ассистента подтверждены данными контекста.
2. **Анализ ответа:**
   - Проверь, не добавляет ли ответ новых фактов, которых нет в контексте.
   - Убедись, что ответ не искажает числа, ставки, сроки и условия.
   - Проверь, не смешивает ли ответ информацию из разных фрагментов, что может привести к неверным утверждениям.
   - Убедись, что ответ покрывает все части вопроса клиента и непосредственно отвечает на заданный вопрос.
3. **Выдача вердикта:** Обрати внимание на независимость оценок по осям. Ошибка по одной оси не должна влиять на оценку по другой оси.

### Формат вывода
- **Reasoning:** Краткий комментарий, объясняющий, почему отданы определённые вердикты.
- **FAITHFULNESS:** PASS или FAIL
- **RELEVANCE:** PASS или FAIL
```
