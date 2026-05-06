#!/bin/bash
# Пример скрипта для запуска с переменными окружения

# Установка переменных окружения
# Скрипт находится в: Active/experimental/Data_Invariants/
# Данные находятся в: Active/experimental/data/ (на уровень выше)
# Если переменные не заданы, используются пути по умолчанию из кода: ../data/raw
# Можно раскомментировать для переопределения:
# export GE_RAW_DATA_PATH=../data/raw
# export GE_CLEAN_DATA_PATH=../data/processed
export GE_DATA_SOURCE_NAME=my_local_datasource
export GE_ASSET_NAME=legal_json_asset
export GE_FILE_PATTERN=**/*.json

# Запуск скрипта
python setup_ge.py

