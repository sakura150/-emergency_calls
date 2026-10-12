import asyncio
import time
import joblib
import polars as pl
import logging
from datetime import datetime
from collections import deque
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()
STOPWORDS_EN = set(stopwords.words('english')) - {
    'other', 'no', 'not', 'off', 'out', 'up', 'down',
    'and', 'or', 'the', 'a', 'an',
}
ABBREVIATIONS = {
    'st': 'street', 'ave': 'avenue', 'rd': 'road',
    'blvd': 'boulevard', 'apt': 'apartment', 'dr': 'drive',
}

def preprocess_text(text: str) -> str:
    """Быстрая предобработка текста"""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'[^a-zA-Z\s]', ' ', text).lower().strip()
    try:
        tokens = word_tokenize(text, language='english')
    except Exception:
        tokens = text.split()
    tokens = [ABBREVIATIONS.get(tok, tok) for tok in tokens]
    lemmas = [
        lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in STOPWORDS_EN and len(tok) > 2
    ]
    return ' '.join(lemmas)


print("Загрузка модели")
try:
    vectorizer = joblib.load("data_splits/tfidf_vectorizer.pkl")
    print("✓ TF-IDF загружен")
except FileNotFoundError:
    print("⚠ Файл tfidf_vectorizer.pkl не найден. Использую заглушку.")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    vectorizer.fit(["hit and run", "fire", "assault weapon", "loud music"])
    print("TF-IDF создан с нуля")

EMERGENCY_KEYWORDS = {
    'assault', 'weapon', 'knife', 'shooting', 'fire',
    'injury', 'robbery', 'homicide', 'rape', 'gun'
}

def classify(text: str) -> dict:
    start = time.time()
    
    text_lower = text.lower()
    is_emergency = any(kw in text_lower for kw in EMERGENCY_KEYWORDS)
    
    confidence = 0.95 if is_emergency else 0.85
    
    latency = (time.time() - start) * 1000  # в мс
    
    return {
        'is_emergency': int(is_emergency),
        'priority': 'Emergency' if is_emergency else 'Non-Emergency',
        'confidence': confidence,
        'latency_ms': latency,
    }


queue = asyncio.Queue(maxsize=1000)   
BATCH_SIZE = 10                        
processing_times = deque(maxlen=100)  
processed_count = 0
emergency_count = 0


async def receive_call(call_id: int, text: str):
    call = {
        'call_id': call_id,
        'timestamp': datetime.now(),
        'text': text,
    }
    await queue.put(call)


async def produce_calls():
    logger.info("Начат приём вызовов")
    
    sample_calls = [
        "ASSAULT WITH WEAPON",
        "LOUD MUSIC COMPLAINT",
        "HIT AND RUN",
        "FIRE IN BUILDING",
        "NARCOTICS",
        "SHOOTING",
        "NOISE COMPLAINT",
        "ROBBERY",
        "AUTO ACCIDENT",
        "KNIFE ATTACK",
        "OTHER",
        "HOMICIDE",
        "DESTRUCT PROP",
        "TRAFFIC STOP",
        "INJURY",
    ]
    
    for i in range(100): 
        text = sample_calls[i % len(sample_calls)]
        await receive_call(i, text)
        await asyncio.sleep(0.1)  
    
    logger.info("Приём вызовов завершён")


async def process_calls():
    global processed_count, emergency_count
    
    logger.info("Запущен обработчик")
    batch = []
    
    while True:
        try:
            call = await asyncio.wait_for(queue.get(), timeout=2.0)
            batch.append(call)
            if len(batch) >= BATCH_SIZE:
                await process_batch(batch)
                batch = []
        
        except asyncio.TimeoutError:
            if batch:
                await process_batch(batch)
                batch = []
            if queue.empty():
                logger.info("Очередь пуста. Обработчик завершает работу.")
                break


async def process_batch(batch):
    global processed_count, emergency_count
    
    batch_start = time.time()
    
    for call in batch:
        clean_text = preprocess_text(call['text'])
        
        result = classify(clean_text)
        
        processing_times.append(result['latency_ms'])
        processed_count += 1
        if result['is_emergency']:
            emergency_count += 1
        
        logger.info(
            f"Вызов #{call['call_id']:3d} | "
            f"{call['text']:25s} | "
            f"→ {result['priority']:15s} | "
            f"conf={result['confidence']:.2f} | "
            f"latency={result['latency_ms']:.1f}ms"
        )
    
    batch_time = (time.time() - batch_start) * 1000
    logger.info(f"  └─ Батч из {len(batch)} вызовов обработан за {batch_time:.1f}мс")


async def main():
    global processed_count, emergency_count
    
    start = time.time()
    
    await asyncio.gather(
        produce_calls(),
        process_calls(),
    )
    
    total_time = time.time() - start
    
    print(f"Всего обработано:    {processed_count} вызовов")
    print(f"Экстренных:          {emergency_count} ({emergency_count/processed_count*100:.1f}%)")
    print(f"Неэкстренных:        {processed_count - emergency_count}")
    print(f"Общее время:         {total_time:.2f} сек")
    print(f"Пропускная способность: {processed_count / total_time:.1f} вызовов/сек")
    
    if processing_times:
        avg_latency = sum(processing_times) / len(processing_times)
        max_latency = max(processing_times)
        print(f"Средняя задержка:    {avg_latency:.1f} мс")
        print(f"Максимальная задержка: {max_latency:.1f} мс")
    


if __name__ == "__main__":
    asyncio.run(main())