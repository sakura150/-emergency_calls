import re
import nltk
import polars as pl
import os
import time
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

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
    'ln': 'lane', 'ct': 'court', 'pl': 'place', 'sq': 'square',
}

def preprocess_text(text: str, lang: str = 'en') -> str:
    if not text or not isinstance(text, str):
        return ""
    
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Zа-яА-ЯёЁ\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    
    try:
        tokens = word_tokenize(text, language='english' if lang == 'en' else 'russian')
    except Exception:
        tokens = text.split()
    
    tokens = [ABBREVIATIONS.get(tok, tok) for tok in tokens]
    
    if lang == 'en':
        lemmas = [
            lemmatizer.lemmatize(tok)
            for tok in tokens
            if tok not in STOPWORDS_EN and len(tok) > 2
        ]
    else:
        import pymorphy3
        morph = pymorphy3.MorphAnalyzer()
        stop_words_ru = set(stopwords.words('russian'))
        lemmas = [
            morph.parse(tok)[0].normal_form
            for tok in tokens
            if tok not in stop_words_ru and len(tok) > 2
        ]
    
    return ' '.join(lemmas)


CSV_PATH = "911_Calls_for_Service.csv"
OUTPUT_PATH = "911_cleaned_all.parquet"

if os.path.exists(OUTPUT_PATH):
    try:
        df = pl.read_parquet(OUTPUT_PATH)
        if df.shape[0] > 0:
            print(f"Файл {OUTPUT_PATH} уже существует. Загружаю...")
            print(f"  Строк: {df.shape[0]}, колонок: {df.shape[1]}")
            print(df.head(10))
            exit(0)
        else:
            print(f"Файл {OUTPUT_PATH} пустой (0 строк). Удаляю и обрабатываю заново...")
            os.remove(OUTPUT_PATH)
    except Exception as e:
        print(f"Не удалось прочитать {OUTPUT_PATH}: {e}")
        print("  Удаляю и обрабатываю заново...")
        os.remove(OUTPUT_PATH)

lf_calls = pl.scan_csv(
    CSV_PATH,
    separator='\t',
    infer_schema_length=10000,
    ignore_errors=True,
)

lf_calls = lf_calls.drop([""], strict=False)

lf_calls = lf_calls.with_columns(
    pl.col('callDateTime')
      .str.to_datetime(format='%Y/%m/%d %H:%M:%S%#z', strict=False)
      .alias('callDateTime')
)


start = time.time()

lf_calls = lf_calls.with_columns(
    pl.col('description')
      .map_batches(
          lambda s: pl.Series(
              [preprocess_text(x, lang='en') for x in s],
              dtype=pl.Utf8,
          ),
          return_dtype=pl.Utf8,
      )
      .alias('description_clean')
)

lf_calls = lf_calls.select([
    'callDateTime', 'priority', 'district',
    'description', 'description_clean',
])


lf_calls.sink_parquet(OUTPUT_PATH)
elapsed = time.time() - start
print(f"Время обработки: {elapsed:.1f} сек")

result = pl.read_parquet(OUTPUT_PATH)

other_check = result.filter(pl.col('description') == 'OTHER').head(5)
print(other_check)
print(result.head(10))