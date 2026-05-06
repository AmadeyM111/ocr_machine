"""Скрипт для чтения и просмотра содержимого .pkl файлов."""

import argparse
import pickle
import pprint
import sys
from pathlib import Path


def read_pkl(filepath: str, max_items: int | None = None):
    path = Path(filepath)
    if not path.exists():
        print(f"Файл не найден: {filepath}")
        sys.exit(1)

    with open(path, "rb") as f:
        data = pickle.load(f)

    print(f"Файл: {filepath}")
    print(f"Тип данных: {type(data).__name__}")

    if isinstance(data, (list, tuple)):
        print(f"Количество элементов: {len(data)}")
        items = data[:max_items] if max_items else data
        for i, item in enumerate(items):
            print(f"\n--- Элемент {i} ---")
            pprint.pprint(item, width=120)
    elif isinstance(data, dict):
        print(f"Ключи ({len(data)}): {list(data.keys())[:20]}")
        keys = list(data.keys())[:max_items] if max_items else list(data.keys())
        for key in keys:
            print(f"\n--- {key} ---")
            pprint.pprint(data[key], width=120, depth=3)
    else:
        pprint.pprint(data, width=120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Чтение .pkl файлов")
    parser.add_argument("filepath", help="Путь к .pkl файлу")
    parser.add_argument(
        "-n", "--max-items", type=int, default=None,
        help="Максимальное количество элементов для отображения",
    )
    args = parser.parse_args()
    read_pkl(args.filepath, args.max_items)
