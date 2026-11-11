import polars as pl
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import recall_score, precision_score, f1_score

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32, 
    device_map="cpu" 
)
model.eval() 
print("Модель загружена")

print("\nЗагрузка тестовых данных ")
df = pl.read_parquet("911_cleaned.parquet")
df = df.filter(
    pl.col('priority').is_not_null() &
    (pl.col('priority') != 'Out of Service')
)
df = df.with_columns(
    pl.when(pl.col('priority').is_in(['Emergency', 'High']))
      .then(1).otherwise(0).alias('is_emergency')
)

test_sample = df.sample(n=30, seed=42)
texts = test_sample['description_clean'].to_list()
labels = test_sample['is_emergency'].to_list()

print(f"Тестовых примеров для LLM: {len(texts)}")

def generate_response(prompt: str) -> str:
    
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=10, 
        do_sample=False,  
        pad_token_id=tokenizer.eos_token_id
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()

def classify_zero_shot(text):
    prompt = f"""Ты — оператор службы 911. Классифицируй вызов как "Emergency" (угроза жизни) или "Non-Emergency" (не требует срочного вмешательства).

Текст вызова: "{text}"

Ответь одним словом: Emergency или Non-Emergency."""
    
    response = generate_response(prompt)
    
    if "Emergency" in response and "Non-Emergency" not in response:
        return 1
    else:
        return 0

def classify_few_shot(text):
    prompt = f"""Ты — оператор службы 911. Классифицируй вызов как "Emergency" или "Non-Emergency".

Примеры:
Текст: "SHOOTING" → Emergency
Текст: "LOUD MUSIC" → Non-Emergency
Текст: "ASSAULT WITH WEAPON" → Emergency

Текст: "{text}"
Ответь одним словом: Emergency или Non-Emergency."""
    
    response = generate_response(prompt)
    
    if "Emergency" in response and "Non-Emergency" not in response:
        return 1
    else:
        return 0

print("\nZero-shot (может занять время на CPU)")
y_pred_zero = []
latencies_zero = []

for i, text in enumerate(texts):
    start = time.time()
    pred = classify_zero_shot(text)
    latency = time.time() - start
    latencies_zero.append(latency)
    y_pred_zero.append(pred)
    print(f"[{i+1}/{len(texts)}] {text[:40]:40s} → {'Emergency' if pred else 'Non-Emergency'} ({latency:.1f}s)")

print(f"\nZero-shot результаты:")
print(f"Recall:    {recall_score(labels, y_pred_zero, pos_label=1, zero_division=0):.4f}")
print(f"Precision: {precision_score(labels, y_pred_zero, pos_label=1, zero_division=0):.4f}")
print(f"F1:        {f1_score(labels, y_pred_zero, pos_label=1, zero_division=0):.4f}")
print(f"Средняя задержка: {sum(latencies_zero)/len(latencies_zero):.2f} сек")

print("\nFew-shot (может занять время на CPU)")
y_pred_few = []
latencies_few = []

for i, text in enumerate(texts):
    start = time.time()
    pred = classify_few_shot(text)
    latency = time.time() - start
    latencies_few.append(latency)
    y_pred_few.append(pred)
    print(f"[{i+1}/{len(texts)}] {text[:40]:40s} → {'Emergency' if pred else 'Non-Emergency'} ({latency:.1f}s)")

print(f"\nFew-shot результаты:")
print(f"Recall:    {recall_score(labels, y_pred_few, pos_label=1, zero_division=0):.4f}")
print(f"Precision: {precision_score(labels, y_pred_few, pos_label=1, zero_division=0):.4f}")
print(f"F1:        {f1_score(labels, y_pred_few, pos_label=1, zero_division=0):.4f}")
print(f"Средняя задержка: {sum(latencies_few)/len(latencies_few):.2f} сек")

results = pl.DataFrame({
    "method": ["Zero-shot", "Few-shot"],
    "recall": [recall_score(labels, y_pred_zero, pos_label=1, zero_division=0),
               recall_score(labels, y_pred_few, pos_label=1, zero_division=0)],
    "precision": [precision_score(labels, y_pred_zero, pos_label=1, zero_division=0),
                  precision_score(labels, y_pred_few, pos_label=1, zero_division=0)],
    "f1": [f1_score(labels, y_pred_zero, pos_label=1, zero_division=0),
           f1_score(labels, y_pred_few, pos_label=1, zero_division=0)],
    "avg_latency_sec": [sum(latencies_zero)/len(latencies_zero), sum(latencies_few)/len(latencies_few)],
})
print("\nИтоговая таблица LLM")
print(results)
results.write_csv("results/llm_local_qwen_comparison.csv")
print("Сохранено: results/llm_local_qwen_comparison.csv")