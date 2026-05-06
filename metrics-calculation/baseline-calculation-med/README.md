## Что делает скрипт

Берёт JSONL-файл с логами диалогов чат-бота (`cummulate_file.jsonl`), извлекает пары вопрос–ответ, аннотирует их и сохраняет результат в 3 файла.

---

## Файлы на выходе и их поля

### 1. `baseline_full.csv` — полный датасет

|Поле|Откуда|
|---|---|
|`conversation_id`|ID диалога из лога|
|`timestamp`|Время сообщения (парсится в datetime)|
|`user_query`|Текст вопроса пользователя|
|`bot_answer`|Текст ответа бота|
|`intent`|Из переменной `intenet` в Event (или `unknown`)|
|`duration`|Время обработки из Event|
|`answer_quality`|Автоавтоматическая оценка: `good`, `bad`, `clarification`, `partial`, `neutral`|
|`intent_category`|Категория: `price`, `doctors`, `location`, `preparation`, `sessions`, `protocol_issues`, `other`|
|`query_length`|Длина вопроса в символах|
|`answer_length`|Длина ответа в символах|
|`has_price_info`|bool — есть ли цена/ОМС|
|`has_phone`|bool — есть ли номер +7...|
|`has_address`|bool — есть ли адрес|

### 2. `baseline_test_set.csv` — тестовая выборка (~50 примеров)

Все поля из `baseline_full.csv` **плюс**:

|Поле|Назначение|
|---|---|
|`overall_quality`|Пустое — для ручной оценки 1–5|
|`reference_answer`|Копия `bot_answer` как ground truth|
|`notes`|Пустое — для комментариев|

### 3. `baseline_test_set.json` — JSON-версия тестовой выборки

Поля: `user_query`, `bot_answer` → переименован в `reference_answer`, `intent`, `intent_category`, `answer_quality`.