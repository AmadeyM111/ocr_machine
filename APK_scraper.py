import requests
import pandas as pd
from bs4 import BeautifulSoup

# --- НАСТРОЙКИ ---
DOC_ID = "470988"  # АПК РФ
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)',
    'Referer': 'http://actual.pravo.gov.ru/'
}

def get_structure(doc_id):
    url = f"http://actual.pravo.gov.ru:8000/api/ebpi/getcontent/?bpa=ebpi&rdk={doc_id}"
    print(f"1. Скачиваю структуру...")
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()

def get_full_text_html(doc_id):
    url = f"http://actual.pravo.gov.ru:8000/api/ebpi/redtext?bpa=ebpi&t={doc_id}&ttl=1"
    print(f"2. Скачиваю текст...")
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()
    return resp.json()['redtext']

def build_dataset():
    # 1. Скачиваем
    try:
        structure_data = get_structure(DOC_ID)
        html_content = get_full_text_html(DOC_ID)
    except Exception as e:
        print(f"Ошибка: {e}")
        return

    # 2. Парсим HTML
    print("3. Индексируем текст...")
    soup = BeautifulSoup(html_content, 'html.parser')
    
    all_paragraphs = []
    para_map = {} 

    for p in soup.find_all('p', id=True):
        p_id = p['id']
        text = p.get_text(strip=True)
        if text: 
            all_paragraphs.append({'id': p_id, 'text': text})
            para_map[p_id] = len(all_paragraphs) - 1

    # 3. Собираем
    print("4. Формируем датасет...")
    dataset = []
    current_part = "Вводная часть" 

    for item in structure_data['data']:
        if item.get('unit') in ['раздел', 'глава']:
            current_part = item['caption']
            continue

        if item.get('unit') == 'статья':
            title = item['caption']
            start_id = item['np']
            end_id = item['npe']

            start_idx = para_map.get(start_id)
            end_idx = para_map.get(end_id)

            if start_idx is not None and end_idx is not None:
                if end_idx < start_idx: end_idx = start_idx
                
                chunk = all_paragraphs[start_idx : end_idx + 1]
                full_text = "\n".join([p['text'] for p in chunk])
                
                dataset.append({
                    'title': title,
                    'text': full_text,
                    'section': current_part
                })

    # --- РАЗДЕЛЕНИЕ НА ДВА ФАЙЛА ---
    df = pd.DataFrame(dataset)

    # ФАЙЛ 1: Чистый датасет для обучения (без лишних колонок)
    clean_filename = "apk_dataset_CLEAN.csv"
    df[['title', 'section', 'text']].to_csv(clean_filename, index=False, encoding='utf-8-sig')
    
    # ФАЙЛ 2: Статистика (без полного текста, чтобы файл был легким)
    stats_df = df[['title', 'section']].copy()
    stats_df['char_count'] = df['text'].str.len()
    stats_df['word_count'] = df['text'].str.split().str.len()
    
    stats_filename = "apk_dataset_STATS.csv"
    stats_df.to_csv(stats_filename, index=False, encoding='utf-8-sig')

    # Вывод в консоль краткой сводки
    print("\n" + "="*30)
    print(" ГОТОВО! ")
    print("="*30)
    print(f"1. Файл для обучения: {clean_filename} (содержит тексты)")
    print(f"2. Файл статистики:   {stats_filename} (содержит цифры)")
    print("-" * 30)
    print(f"Всего статей: {len(df)}")
    print(f"Средняя длина: {int(stats_df['char_count'].mean())} символов")

if __name__ == "__main__":
    build_dataset()
