# Data Invariants с Great Expectations

Проект для настройки и валидации данных с использованием Great Expectations.

## Возможности

- ✅ Конфигурация через переменные окружения
- ✅ Обработка ошибок с подробным логированием
- ✅ Автоматическая валидация данных
- ✅ Создание и выполнение expectations
- ✅ Использование pathlib.Path для работы с путями

## Установка

```bash
pip install great-expectations
# Опционально, для поддержки .env файлов:
pip install python-dotenv
```

## Настройка переменных окружения

Есть несколько способов задать переменные окружения:

### Способ 1: Файл `.env` (рекомендуется)

1. Создайте файл `.env` в корне проекта:
```bash
touch .env
```

2. Добавьте переменные в `.env` файл:
```bash
GE_RAW_DATA_PATH=../experimental/data/raw
GE_CLEAN_DATA_PATH=../experimental/data/processed
GE_DATA_SOURCE_NAME=my_local_datasource
GE_ASSET_NAME=legal_json_asset
GE_BATCH_DEFINITION_NAME=whole_file
GE_FILE_PATTERN=**/*.json
```

### Работа с данными, разбитыми по классам/кодексам

Если ваши данные организованы в поддиректориях по классам и типам кодексов, используйте паттерн `GE_FILE_PATTERN`:

**Пример структуры данных:**
```
data/raw/
  ├── class1/
  │   ├── codec1/
  │   │   └── data.json
  │   └── codec2/
  │       └── data.json
  ├── class2/
  │   └── codec3/
  │       └── data.json
```

**Настройка в `.env`:**
```bash
# Путь к корневой директории с данными
GE_RAW_DATA_PATH=../experimental/data/raw

# Паттерн для поиска всех JSON файлов рекурсивно во всех поддиректориях
GE_FILE_PATTERN=**/*.json

# Или более специфичный паттерн для определенных классов/кодексов:
# GE_FILE_PATTERN=class*/codec*/*.json
# GE_FILE_PATTERN=class1/**/*.json  # только файлы из class1
```

**Примеры паттернов:**
- `**/*.json` - все JSON файлы рекурсивно во всех поддиректориях
- `class*/*.json` - JSON файлы в директориях, начинающихся с "class"
- `*/codec*/*.json` - JSON файлы в поддиректориях codec* внутри любых директорий
- `class1/**/*.json` - все JSON файлы внутри class1 и его поддиректорий

**Важно:** Файл `.env` должен быть в той же директории, что и `setup_ge.py`

### Способ 2: В терминале перед запуском

```bash
export GE_RAW_DATA_PATH=.
export GE_CLEAN_DATA_PATH=../data/processed
python setup_ge.py
```

### Способ 3: В одной команде

```bash
GE_RAW_DATA_PATH=. GE_CLEAN_DATA_PATH=../data/processed python setup_ge.py
```

### Способ 4: В shell скрипте

Создайте файл `run.sh`:
```bash
#!/bin/bash
export GE_RAW_DATA_PATH=.
export GE_CLEAN_DATA_PATH=../data/processed
python setup_ge.py
```

Затем запустите:
```bash
chmod +x run.sh
./run.sh
```

### Способ 5: Постоянно в системе (не рекомендуется)

Добавьте в `~/.zshrc` или `~/.bashrc`:
```bash
export GE_RAW_DATA_PATH=/your/path/to/data/raw
export GE_CLEAN_DATA_PATH=/your/path/to/data/processed
```

**Примечание:** Если переменные окружения не заданы, используются относительные пути по умолчанию от корня проекта.

## Использование

Запустите скрипт:
```bash
python setup_ge.py
```

### Пример: Работа с данными по классам и кодексам

Если ваши данные организованы так:
```
data/raw/
  ├── criminal/
  │   ├── h264/
  │   │   └── case1.json
  │   └── h265/
  │       └── case2.json
  └── civil/
      └── vp9/
          └── case3.json
```

Настройте `.env`:
```bash
GE_RAW_DATA_PATH=../data/raw
GE_FILE_PATTERN=**/*.json
```

Или для конкретного класса:
```bash
GE_RAW_DATA_PATH=../data/raw
GE_FILE_PATTERN=criminal/**/*.json
```

Скрипт автоматически найдет и обработает все файлы, соответствующие паттерну.

## Структура

- **Context**: Мозг системы (хранит настройки)
- **Data Source**: Указывает на папку с данными
- **Data Asset**: Указывает на конкретный тип файла в этой папке
- **Batch**: Конкретные данные, которые будут проверяться тестами
- **Expectation Suite**: Набор правил для проверки данных
- **Validator**: Объект для выполнения проверок на данных
- **Checkpoint**: Точка проверки, объединяющая данные и expectations

## Логирование

Все операции логируются с уровнем INFO. Логи включают:
- Инициализацию компонентов
- Создание источников данных и assets
- Добавление expectations
- Результаты валидации

## Обработка ошибок

Код включает обработку ошибок на всех этапах:
- Проверка существования директорий
- Обработка ошибок инициализации Great Expectations
- Обработка ошибок создания expectations
- Обработка ошибок валидации

