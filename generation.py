import polars as pl
import random
from datetime import datetime, timedelta
import os

random.seed(42)

TEMPLATES = {
    'Emergency': {
        'weight': 0.11,
        'texts': [
            "Помогите, человек задыхается!",
            "Напали с ножом, кровь!",
            "Пожар в доме, люди внутри!",
            "Убивают, срочно приезжайте!",
            "Человек без сознания, не дышит!",
            "Стрельба во дворе, есть раненые!",
            "Женщина кричит, её избивают!",
            "Ребёнок выпал из окна!",
            "ДТП с пострадавшими, нужна скорая!",
            "Мужчина с пистолетом угрожает людям!",
            "Утопает человек в реке, помогите!",
            "Сильное кровотечение, нужна помощь!",
            "Кто-то ворвался в квартиру с оружием!",
            "Самоубийство на глазах, срочно!",
            "Эпилептический приступ, человеку плохо!",
        ],
    },
    'High': {
        'weight': 0.10,
        'texts': [
            "Драка во дворе, слышны крики",
            "Кто-то ломится в дверь",
            "Женщина кричит о помощи из квартиры",
            "Мужчина угрожает соседям",
            "Подозрительный человек с ножом во дворе",
            "Слышны выстрелы в соседнем доме",
            "Разбито окно, кто-то пытается залезть",
            "Угрожают по телефону, боюсь за жизнь",
            "Погоня, машина врезалась в столб",
            "Группа людей избивает одного",
        ],
    },
    'Medium': {
        'weight': 0.30,
        'texts': [
            "Шумные соседи, мешают спать",
            "Семейный скандал без насилия",
            "Пьяная компания во дворе",
            "Собака лает всю ночь",
            "Музыка громкая у соседей сверху",
            "Беспризорные подростки у подъезда",
            "Машина сигналит без причины",
            "Мусор раскидали у подъезда",
            "Драка собак во дворе",
            "Сосед курит на балконе, дым идёт в квартиру",
        ],
    },
    'Low': {
        'weight': 0.30,
        'texts': [
            "Как вызвать сантехника?",
            "Уточнить адрес поликлиники",
            "Где ближайшая аптека круглосуточно?",
            "Как получить справку о составе семьи?",
            "Не работает лифт в подъезде",
            "Как записаться к врачу?",
            "Разбита лампочка в подъезде",
            "Где находится МФЦ?",
            "Как оплатить коммунальные услуги?",
            "Потерялся ключ от квартиры",
        ],
    },
    'Non-Emergency': {
        'weight': 0.19,
        'texts': [
            "Сообщение о потерянном кошельке",
            "Нашёл чужой паспорт, куда отнести?",
            "Проверить, не разыскивается ли человек",
            "Консультация по штрафу ГИБДД",
            "Уточнить график работы участкового",
            "Жалоба на управляющую компанию",
            "Как подать заявление в полицию?",
            "Вопрос про регистрацию автомобиля",
            "Консультация по административному делу",
            "Уточнить номер телефона отдела",
        ],
    },
}

def generate_call(call_id: int, base_time: datetime) -> dict:
   
    classes = list(TEMPLATES.keys())
    weights = [TEMPLATES[c]['weight'] for c in classes]
    priority = random.choices(classes, weights=weights)[0]
    
    text = random.choice(TEMPLATES[priority]['texts'])
    
    offset = timedelta(seconds=random.randint(-1800, 1800))
    timestamp = base_time + offset
    
    district = random.choice(['NE', 'ND', 'CD', 'SE', 'SW', 'NW', 'TRU'])
    
    return {
        'call_id': call_id,
        'timestamp': timestamp,
        'priority': priority,
        'is_emergency': 1 if priority in ['Emergency', 'High'] else 0,
        'text': text,
        'district': district,
    }

print("Генерация синтетических русскоязычных данных")

N_CALLS = 10_000 
base_time = datetime(2026, 1, 1, 12, 0, 0)

data = [generate_call(i, base_time) for i in range(N_CALLS)]
df_ru = pl.DataFrame(data)

print(f"Сгенерировано {N_CALLS} вызовов")
print()

print("Распределение по приоритетам")
dist = (
    df_ru.group_by('priority')
         .agg(pl.len().alias('count'))
         .with_columns((pl.col('count') / N_CALLS * 100).round(2).alias('share_%'))
         .sort('count', descending=True)
)
print(dist)
print()

OUTPUT_PATH = "ru_synthetic_calls.parquet"
df_ru.write_parquet(OUTPUT_PATH)
print()

print("Примеры вызовов по классам")
for priority in ['Emergency', 'High', 'Medium', 'Low', 'Non-Emergency']:
    sample = df_ru.filter(pl.col('priority') == priority).head(3)
    print(f"\n--- {priority} ---")
    for row in sample.iter_rows(named=True):
        print(f"  [{row['timestamp'].strftime('%H:%M')}] {row['text']}")