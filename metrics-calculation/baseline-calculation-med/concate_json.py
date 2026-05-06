#!/usr/bin/env python3
"""
Конвертирование 73 JSON файлов в один JSONL.
Использование: python convert.py
"""
import json
import glob

def main():
    input_pattern = "*.json"  # Измените на ваш паттерн
    output_file = "cummulate_file.jsonl"
    
    json_files = glob.glob(input_pattern)
    print(f"📁 Найдено файлов: {len(json_files)}")
    
    total = 0
    with open(output_file, 'w', encoding='utf-8') as out:
        for i, file in enumerate(json_files, 1):
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    records = data if isinstance(data, list) else [data]
                    for record in records:
                        json.dump(record, out, ensure_ascii=False)
                        out.write('\n')
                        total += 1
                print(f"✅ [{i}/{len(json_files)}] {file}")
            except Exception as e:
                print(f"❌ {file}: {e}")
    
    print(f"\n✅ Готово! Записей: {total}, Файл: {output_file}")

if __name__ == "__main__":
    main()
