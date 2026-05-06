"""
Скрипт для подготовки baseline из логов диалогов.
"""

import json
import re
import pandas as pd
from typing import List, Dict
from datetime import datetime


def extract_qa_pairs(jsonl_file: str) -> List[Dict]:
    """
    Извлекает пары вопрос-ответ из JSONL логов.
    
    Returns:
        List[Dict]: Список словарей с полями:
            - conversation_id: ID диалога
            - timestamp: Время вопроса
            - user_query: Вопрос пользователя
            - bot_answer: Ответ бота
            - intent: Намерение (если есть в логах)
            - duration: Время обработки
    """
    
    qa_pairs = []
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            
            conv = json.loads(line)
            conv_id = conv.get('id', 'unknown')
            history = conv.get('history', [])
            
            # Извлекаем пары вопрос-ответ
            current_question = None
            current_timestamp = None
            intent = None
            duration = None
            
            for i, msg in enumerate(history):
                msg_type = msg.get('type')
                
                # Пользовательский вопрос
                if msg_type == 'Message' and msg.get('from', {}).get('role') == 'User':
                    current_question = msg.get('text', '').strip()
                    current_timestamp = msg.get('timestamp')
                
                # Извлекаем intent из Event (если есть)
                elif msg_type == 'Event' and 'variables' in msg:
                    variables = msg.get('variables', {})
                    if 'intenet' in variables:  # Да, в логах опечатка: intenet вместо intent
                        intent = variables['intenet']
                    if 'duration' in msg:
                        duration = msg['duration']
                
                # Ответ бота
                elif msg_type == 'Message' and msg.get('from', {}).get('role') == 'Bot':
                    bot_answer = msg.get('text', '').strip()
                    
                    # Пропускаем приветственные сообщения
                    if 'Добро пожаловать' in bot_answer or 'Задайте ваш вопрос' in bot_answer:
                        continue
                    
                    # Пропускаем системные сообщения
                    if 'Рад был помочь' in bot_answer or bot_answer == 'Не указано':
                        continue
                    
                    # Если есть текущий вопрос — создаём пару
                    if current_question:
                        qa_pairs.append({
                            'conversation_id': conv_id,
                            'timestamp': current_timestamp,
                            'user_query': current_question,
                            'bot_answer': bot_answer,
                            'intent': intent or 'unknown',
                            'duration': duration,
                        })
                        
                        # Сбрасываем для следующего вопроса
                        current_question = None
                        intent = None
                        duration = None
    
    return qa_pairs


def categorize_answer_quality(answer: str) -> str:
    """
    Автоматически оценивает качество ответа.

    Returns:
        'good' | 'bad' | 'clarification' | 'partial' | 'neutral'
    """

    answer_lower = answer.lower()

    # Плохие ответы (ошибки, нет информации)
    bad_patterns = [
        'не указано',
        'нет информации',
        'нет конкретной информации',
        'ошибка запроса',
        'не удалось получить',
    ]

    if any(pattern in answer_lower for pattern in bad_patterns):
        return 'bad'

    # Уточняющие вопросы
    clarification_patterns = [
        'уточните',
        'какая конкретная услуга',
        'конкретно',
        'какой именно',
    ]

    if any(pattern in answer_lower for pattern in clarification_patterns):
        return 'clarification'

    # Признаки содержательного ответа (конкретная информация)
    content_patterns = [
        'рубл', 'руб', '₽', 'ОМС', 'бесплатно',  # цена
        '+7', 'тел',                                 # контакты
        'адрес', 'москва', 'химки', 'клязьма',      # адрес
        'врач', 'доктор', 'специалист',              # специалисты
        'приём', 'запись', 'расписание',             # запись
    ]
    has_content = any(p in answer_lower for p in content_patterns)

    # Хороший ответ: длинный И содержит конкретику
    if len(answer) > 100 and has_content:
        return 'good'

    # Частичный ответ: есть конкретика, но с оговоркой позвонить
    partial_patterns = ['рекомендую обратиться', 'позвоните']
    if any(p in answer_lower for p in partial_patterns):
        return 'partial'

    # Хороший ответ: длинный, без оговорок (даже без явных content_patterns)
    if len(answer) > 100:
        return 'good'

    return 'neutral'


def annotate_intent_category(intent: str) -> str:
    """
    Категоризирует intent по типам запросов.
    
    Returns:
        'price' | 'doctors' | 'location' | 'preparation' | 'other'
    """
    
    intent_lower = intent.lower()
    
    if 'цен' in intent_lower or 'стоимост' in intent_lower:
        return 'price'
    elif 'врач' in intent_lower or 'доктор' in intent_lower:
        return 'doctors'
    elif 'адрес' in intent_lower or 'филиал' in intent_lower:
        return 'location'
    elif 'подготовк' in intent_lower or 'анализ' in intent_lower:
        return 'preparation'
    elif 'сеанс' in intent_lower:
        return 'sessions'
    elif 'протокол' in intent_lower or 'ошибк' in intent_lower:
        return 'protocol_issues'
    else:
        return 'other'


def create_baseline_dataset(qa_pairs: List[Dict]) -> pd.DataFrame:
    """
    Создаёт итоговый baseline датасет.
    """
    
    # Преобразуем в DataFrame
    df = pd.DataFrame(qa_pairs)
    
    # Добавляем автоматическую оценку качества
    df['answer_quality'] = df['bot_answer'].apply(categorize_answer_quality)
    
    # Категоризируем intent
    df['intent_category'] = df['intent'].apply(annotate_intent_category)
    
    # Добавляем метаданные
    df['query_length'] = df['user_query'].str.len()
    df['answer_length'] = df['bot_answer'].str.len()
    df['has_price_info'] = df['bot_answer'].str.contains('рубл|руб|₽|ОМС|бесплатно', case=False, regex=True)
    df['has_phone'] = df['bot_answer'].str.contains(r'\+7\s*\(?\d{3}\)?', regex=True)
    df['has_address'] = df['bot_answer'].str.contains('Москва|Химки|Клязьма|Тверская', case=False, regex=True)
    
    # Парсим timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


def generate_test_set(df: pd.DataFrame, sample_size: int = 50) -> pd.DataFrame:
    """
    Генерирует тестовый набор для оценки.
    
    Стратифицированная выборка по intent_category и answer_quality.
    """
    
    # Берём только 'good' и 'partial' ответы для базового теста
    test_df = df[df['answer_quality'].isin(['good', 'partial'])].copy()
    
    # Стратифицированная выборка
    if len(test_df) > sample_size:
        test_df = test_df.groupby('intent_category', group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, int(sample_size * len(x) / len(test_df)))))
        )
    
    # Добавляем поле для ручной аннотации
    test_df['overall_quality'] = ''  # Заполнить вручную: 1-5
    test_df['reference_answer'] = test_df['bot_answer']  # Ground truth
    test_df['notes'] = ''  # Комментарии
    
    return test_df.reset_index(drop=True)


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    
    print("=" * 60)
    print(" ПОДГОТОВКА BASELINE ИЗ ЛОГОВ ДИАЛОГОВ")
    print("=" * 60)
    
    # 1. Извлечение пар вопрос-ответ
    print("\n[1/5] Извлечение пар вопрос-ответ...")
    qa_pairs = extract_qa_pairs('cummulate_file.jsonl')
    print(f"✅ Извлечено пар: {len(qa_pairs)}")
    
    # 2. Создание датасета
    print("\n[2/5] Создание датасета с метаданными...")
    df = create_baseline_dataset(qa_pairs)
    print(f"✅ Создано записей: {len(df)}")
    
    # Статистика
    print("\n📊 СТАТИСТИКА ДАТАСЕТА:")
    print(f"Всего пар вопрос-ответ: {len(df)}")
    print(f"\nРаспределение по качеству ответов:")
    print(df['answer_quality'].value_counts())
    print(f"\nРаспределение по категориям intent:")
    print(df['intent_category'].value_counts())
    
    # 3. Сохранение полного датасета
    print("\n[3/5] Сохранение полного датасета...")
    df.to_csv('baseline_full.csv', index=False, encoding='utf-8-sig')
    print("✅ Сохранено: baseline_full.csv")
    
    # 4. Генерация тестового набора
    print("\n[4/5] Генерация тестового набора (50 примеров)...")
    test_df = generate_test_set(df, sample_size=50)
    test_df.to_csv('baseline_test_set.csv', index=False, encoding='utf-8-sig')
    print("✅ Сохранено: baseline_test_set.csv")
    
    # 5. Сохранение JSON для использования в коде
    print("\n[5/5] Сохранение JSON версии...")

    json_columns = ['user_query', 'bot_answer', 'intent', 'intent_category', 'answer_quality']
    test_json_df = test_df[json_columns].rename(columns={'bot_answer': 'reference_answer'})
    test_json = test_json_df.to_dict('records')

    with open('baseline_test_set.json', 'w', encoding='utf-8') as f:
        json.dump(test_json, f, ensure_ascii=False, indent=2)
    print("✅ Сохранено: baseline_test_set.json")
    
    print("\n" + "=" * 60)
    print("✅ BASELINE ГОТОВ!")
    print("=" * 60)
    print("\nСледующие шаги:")
    print("1. Откройте baseline_test_set.csv")
    print("2. Заполните колонку 'overall_quality' (1-5)")
    print("3. Используйте для A/B тестирования")
