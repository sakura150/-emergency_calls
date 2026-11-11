import polars as pl
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
import joblib
import os

CLEANED_PATH = "911_cleaned_all.parquet"
df = pl.read_parquet(CLEANED_PATH)
print(f"Строк: {df.shape[0]}")
print(f"Колонки: {df.columns}")
print()

df = df.filter(
    pl.col('priority').is_not_null() &
    (pl.col('priority') != 'Out of Service')
)

df = df.with_columns(
    pl.when(pl.col('priority').is_in(['Emergency', 'High']))
      .then(1)
      .otherwise(0)
      .alias('is_emergency')
)

print("Распределение классов ДО балансировки")
before = (
    df.group_by('is_emergency')
      .agg(pl.len().alias('count'))
      .sort('is_emergency', descending=True)
)
total = before['count'].sum()
before = before.with_columns(
    (pl.col('count') / total * 100).round(2).alias('share_%')
)
print(before)
print()

count_0 = before.filter(pl.col('is_emergency') == 0)['count'].item()
count_1 = before.filter(pl.col('is_emergency') == 1)['count'].item()
print(f"Соотношение класс 0 : класс 1 = {count_0 / count_1:.1f} : 1")
print()

print("Векторизация TF-IDF")
texts = df['description_clean'].to_list()
labels = df['is_emergency'].to_numpy()

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),  
    min_df=2,         
)
X = vectorizer.fit_transform(texts)
y = labels

print(f"Размер матрицы признаков: {X.shape}")
print(f"Количество признаков: {X.shape[1]}")
print()

print("Применение SMOTE")
minority_count = np.sum(y == 1)
k = min(5, minority_count - 1)

smote = SMOTE(random_state=42, k_neighbors=k)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"Размер до SMOTE: {X.shape}")
print(f"Размер после SMOTE: {X_resampled.shape}")
print()

print("Распределение классов ПОСЛЕ балансировки")
after = pl.DataFrame({
    'is_emergency': y_resampled,
}).group_by('is_emergency').agg(pl.len().alias('count')).sort('is_emergency', descending=True)

total_after = after['count'].sum()
after = after.with_columns(
    (pl.col('count') / total_after * 100).round(2).alias('share_%')
)
print(after)
print()

from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(
    X_resampled, y_resampled,
    test_size=0.3, stratify=y_resampled, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5, stratify=y_temp, random_state=42
)

print("Размеры выборок")
print(f"Train: {X_train.shape[0]}")
print(f"Val:   {X_val.shape[0]}")
print(f"Test:  {X_test.shape[0]}")
print()

OUTPUT_DIR = "data_splits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

joblib.dump((X_train, y_train), f"{OUTPUT_DIR}/train.pkl")
joblib.dump((X_val, y_val), f"{OUTPUT_DIR}/val.pkl")
joblib.dump((X_test, y_test), f"{OUTPUT_DIR}/test.pkl")
joblib.dump(vectorizer, f"{OUTPUT_DIR}/tfidf_vectorizer.pkl")

