import joblib
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Загрузка данных")
X_train, y_train = joblib.load("data_splits/train_fixed.pkl")
X_val, y_val = joblib.load("data_splits/val_fixed.pkl")
X_test, y_test = joblib.load("data_splits/test_fixed.pkl")

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print()

models = {
    "Logistic Regression": LogisticRegression(
        class_weight='balanced',
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
    ),
    "SVM (LinearSVC)": LinearSVC(
        class_weight='balanced',
        random_state=42,
        dual='auto',
        max_iter=2000,
    ),
    "LightGBM": lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    ),
}

results = []

for name, model in models.items():
    print(f"Обучение: {name}")
    
    model.fit(X_train, y_train)
    
    y_pred_val = model.predict(X_val)
    
    if hasattr(model, "predict_proba"):
        y_score_val = model.predict_proba(X_val)[:, 1]
    else: 
        y_score_val = model.decision_function(X_val)
    
    recall = recall_score(y_val, y_pred_val, pos_label=1)
    precision = precision_score(y_val, y_pred_val, pos_label=1)
    f1 = f1_score(y_val, y_pred_val, pos_label=1)
    roc_auc = roc_auc_score(y_val, y_score_val)
    
    results.append({
        "model": name,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "roc_auc": roc_auc,
    })
    
    print(f"Recall (Emergency):    {recall:.4f}")
    print(f"Precision (Emergency): {precision:.4f}")
    print(f"F1 (Emergency):        {f1:.4f}")
    print(f"ROC-AUC:               {roc_auc:.4f}")
    print()
    print(classification_report(y_val, y_pred_val, target_names=['Non-Emergency', 'Emergency']))
   

print("\n=== Оценка на тестовой выборке ===")

best_result = max(results, key=lambda x: x["recall"])
best_model_name = best_result["model"]
best_model = models[best_model_name]

print(f"Лучшая модель по Recall: {best_model_name}")

y_pred_test = best_model.predict(X_test)
if hasattr(best_model, "predict_proba"):
    y_score_test = best_model.predict_proba(X_test)[:, 1]
else:
    y_score_test = best_model.decision_function(X_test)

print("\nClassification Report (Test):")
print(classification_report(y_test, y_pred_test, target_names=['Non-Emergency', 'Emergency']))
print(f"ROC-AUC (Test): {roc_auc_score(y_test, y_score_test):.4f}")

os.makedirs("results", exist_ok=True)

cm = confusion_matrix(y_test, y_pred_test)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Non-Emergency', 'Emergency'],
    yticklabels=['Non-Emergency', 'Emergency'],
    ax=ax,
)
ax.set_xlabel('Предсказано')
ax.set_ylabel('Истинно')
ax.set_title(f'Матрица ошибок: {best_model_name}')
plt.tight_layout()
plt.savefig("results/confusion_matrix.png", dpi=150)
plt.close()
print("\nМатрица ошибок сохранена: results/confusion_matrix.png")

df_results = pl.DataFrame(results)
df_results.write_csv("results/classic_models_comparison.csv")
print("Таблица результатов сохранена: results/classic_models_comparison.csv")
print()
print(df_results)

joblib.dump(best_model, f"results/best_classic_model_{best_model_name.replace(' ', '_')}.pkl")
print(f"Лучшая модель сохранена: results/best_classic_model_{best_model_name.replace(' ', '_')}.pkl")