import logging
from pathlib import Path
from pdf_ocr_advanced import extract_text_advanced

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

if __name__ == "__main__":
    pdf_file = Path("input.pdf")
    if not pdf_file.exists():
        raise FileNotFoundError(f"Положите тестируемый PDF как {pdf_file}")

    # Вызов с параметрами по умолчанию + переопределение на лету
    text = extract_text_advanced(
        pdf_path=str(pdf_file),
        lang="rus+eng",
        dpi=400,
        enable_preprocessing=True,
        enable_postprocessing=True,
        num_workers=-1
    )

    # Сохранение результата
    output_path = Path("output.txt")
    output_path.write_text(text, encoding="utf-8")
    
    logging.info(f"Готово. Распознано {len(text)} символов. Сохранено в {output_path}")
    print("\n--- ПЕРВЫЕ 300 СИМВОЛОВ ---")
    print(text[:300])