import joblib
import numpy as np
import polars as pl
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score, roc_auc_score
import time

print("Загрузка текстов")

df = pl.read_parquet("911_cleaned.parquet")
df = df.filter(
    pl.col('priority').is_not_null() &
    (pl.col('priority') != 'Out of Service')
)
df = df.with_columns(
    pl.when(pl.col('priority').is_in(['Emergency', 'High']))
      .then(1).otherwise(0).alias('is_emergency')
)

texts = df['description_clean'].to_list()
labels = df['is_emergency'].to_numpy()

print(f"Всего текстов: {len(texts)}")

from sklearn.model_selection import train_test_split

X_train_text, X_temp_text, y_train, y_temp = train_test_split(
    texts, labels, test_size=0.3, stratify=labels, random_state=42
)
X_val_text, X_test_text, y_val, y_test = train_test_split(
    X_temp_text, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

print(f"Train: {len(X_train_text)}, Val: {len(X_val_text)}, Test: {len(X_test_text)}")

print("\nЗагрузка Sentence Transformer")
start = time.time()
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f"Модель загружена за {time.time() - start:.1f} сек")

print("\nГенерация эмбеддингов (CPU)")

start = time.time()
X_train_emb = model.encode(X_train_text, batch_size=64, show_progress_bar=True)
print(f"Train embeddings: {X_train_emb.shape}, время: {time.time() - start:.1f} сек")

start = time.time()
X_val_emb = model.encode(X_val_text, batch_size=64, show_progress_bar=True)
print(f"Val embeddings: {X_val_emb.shape}, время: {time.time() - start:.1f} сек")

start = time.time()
X_test_emb = model.encode(X_test_text, batch_size=64, show_progress_bar=True)
print(f"Test embeddings: {X_test_emb.shape}, время: {time.time() - start:.1f} сек")

from imblearn.over_sampling import SMOTE

print("\nSMOTE на эмбеддингах")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train_emb, y_train)

print(f"Train до SMOTE: {X_train_emb.shape}")
print(f"Train после SMOTE: {X_train_res.shape}")

models = {
    "LogReg + SBERT": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
    "SVM + SBERT": LinearSVC(class_weight='balanced', random_state=42, dual='auto', max_iter=2000),
}

results = []

for name, clf in models.items():
    print(f"\n{name}")
    
    clf.fit(X_train_res, y_train_res)
    y_pred = clf.predict(X_val_emb)
    
    if hasattr(clf, "predict_proba"):
        y_score = clf.predict_proba(X_val_emb)[:, 1]
    else:
        y_score = clf.decision_function(X_val_emb)
    
    recall = recall_score(y_val, y_pred, pos_label=1)
    precision = precision_score(y_val, y_pred, pos_label=1)
    f1 = f1_score(y_val, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_val, y_score)
    
    results.append({"model": name, "recall": recall, "precision": precision, "f1": f1, "roc_auc": roc_auc})
    
    print(f"Recall:    {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(classification_report(y_val, y_pred, target_names=['Non-Emergency', 'Emergency']))


print("\nИтоговая таблица")
df_results = pl.DataFrame(results)
print(df_results)

# Сохраняем
df_results.write_csv("results/sbert_models_comparison.csv")
print("\nСохранено: results/sbert_models_comparison.csv")