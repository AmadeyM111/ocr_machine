import json

INPUT_PATH = "gk_rf_articles.json"       # файл с текущим выводом скрипта
OUTPUT_PATH = "gk_postprocessed.json"

def strip_title_from_content(item):
    text = item.get("content", "")
    lines = text.splitlines()

    # если первая строка начинается с "Статья" — считаем её заголовком и убираем
    if lines and lines[0].strip().startswith("Статья"):
        lines = lines[1:]

    new_text = "\n".join(l for l in lines).lstrip("\n")

    item["content"] = new_text
    return item

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)  # ожидаем список объектов

    processed = []
    for obj in data:
        obj = strip_title_from_content(obj)

        # если хочешь выбросить служебные поля — делай это здесь
        # obj.pop("header_id", None)

        processed.append(obj)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"Обработано объектов: {len(processed)}")

if __name__ == "__main__":
    main()
