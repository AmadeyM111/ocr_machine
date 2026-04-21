# DS_Tools

## data scientist toolkit

# OCR

ocr_pdf.py

# 1. Проверка системных зависимостей
tesseract --version
pdfinfo -v

# 2. Активация venv
source env/bin/activate  # Linux/Mac
# или
env\Scripts\activate     # Windows

# 3. Установка Python-пакетов
pip install -r requirements.txt

# 4. Создание .env (если нет)
cat > .env << EOF
OCR_TESSERACT_CMD=tesseract
OCR_DEFAULT_LANG=rus+eng
OCR_DPI=400
OCR_ENABLE_PREPROCESSING=true
OCR_ENABLE_POSTPROCESSING=true
EOF

# 5. Тестовый запуск
python main.py