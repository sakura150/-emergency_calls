import polars as pl
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import os

CLEANED_PATH = "911_cleaned_all.parquet"
df = pl.read_parquet(CLEANED_PATH)
print(f"Строк: {df.shape[0]}")
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

OUTPUT_DIR = "eda_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Распределение классов")
counts = df.group_by('is_emergency').agg(pl.len().alias('count')).sort('is_emergency')
print(counts)

fig, ax = plt.subplots(figsize=(6, 4))
labels = ['Non-Emergency (0)', 'Emergency (1)']
values = counts['count'].to_list()
colors = ['#4C72B0', '#C44E52']
bars = ax.bar(labels, values, color=colors)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f'{val:,}', ha='center', va='bottom', fontsize=11)

ax.set_ylabel('Количество вызовов')
ax.set_title('Распределение классов в датасете')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/class_distribution.png', dpi=150)
plt.close()
print(f"Сохранено: {OUTPUT_DIR}/class_distribution.png")
print()


print("Частотный анализ слов")

emergency_texts = df.filter(pl.col('is_emergency') == 1)['description_clean'].to_list()
non_emergency_texts = df.filter(pl.col('is_emergency') == 0)['description_clean'].to_list()

emergency_words = Counter(" ".join(emergency_texts).split())
non_emergency_words = Counter(" ".join(non_emergency_texts).split())

top_emergency = emergency_words.most_common(20)
top_non_emergency = non_emergency_words.most_common(20)

print("\nТОП-20 слов в экстренных вызовах вызовах:")
for word, count in top_emergency:
    print(f"  {word:20s} {count}")

print("\nТОП-20 слов в неэкстренных вызовах:")
for word, count in top_non_emergency:
    print(f"  {word:20s} {count}")
print()

fig, ax = plt.subplots(figsize=(10, 6))
words, counts = zip(*top_emergency)
ax.barh(range(len(words)), counts, color='#C44E52')
ax.set_yticks(range(len(words)))
ax.set_yticklabels(words)
ax.invert_yaxis()
ax.set_xlabel('Частота')
ax.set_title('Топ-20 слов в экстренных вызовах')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/top20_emergency.png', dpi=150)
plt.close()
print(f"Сохранено: {OUTPUT_DIR}/top20_emergency.png")

fig, ax = plt.subplots(figsize=(10, 6))
words, counts = zip(*top_non_emergency)
ax.barh(range(len(words)), counts, color='#4C72B0')
ax.set_yticks(range(len(words)))
ax.set_yticklabels(words)
ax.invert_yaxis()
ax.set_xlabel('Частота')
ax.set_title('Топ-20 слов в неэкстренных вызовах')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/top20_non_emergency.png', dpi=150)
plt.close()
print(f"Сохранено: {OUTPUT_DIR}/top20_non_emergency.png")

print("\nОблако слов")

emergency_text_joined = " ".join(emergency_texts)
wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color='white',
    colormap='Reds',
    max_words=100,
).generate(emergency_text_joined)

fig, ax = plt.subplots(figsize=(14, 7))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
ax.set_title('Облако слов для экстренных вызовов', fontsize=16)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/wordcloud_emergency.png', dpi=150)
plt.close()
print(f"Сохранено: {OUTPUT_DIR}/wordcloud_emergency.png")

non_emergency_text_joined = " ".join(non_emergency_texts)
wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color='white',
    colormap='Blues',
    max_words=100,
).generate(non_emergency_text_joined)

fig, ax = plt.subplots(figsize=(14, 7))
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
ax.set_title('Облако слов для неэкстренных вызовов', fontsize=16)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/wordcloud_non_emergency.png', dpi=150)
plt.close()
print(f"Сохранено: {OUTPUT_DIR}/wordcloud_non_emergency.png")

print("\nКлючевые маркеры экстренности")

total_emergency = sum(emergency_words.values())
total_non_emergency = sum(non_emergency_words.values())

markers = []
for word in emergency_words:
    freq_emerg = emergency_words[word] / total_emergency
    freq_non = non_emergency_words.get(word, 0) / total_non_emergency
    if freq_non > 0 and freq_emerg > 0.001:  # слово встречается не реже 0.1%
        ratio = freq_emerg / freq_non
        markers.append((word, emergency_words[word], non_emergency_words.get(word, 0), ratio))

markers.sort(key=lambda x: x[3], reverse=True)
print("\nТоп-15 маркеров экстренности (по отношению частот):")
print(f"{'Слово':20s} {'Экстр':>8s} {'Неэкстр':>10s} {'Ratio':>8s}")
for word, e, ne, r in markers[:15]:
    print(f"{word:20s} {e:>8d} {ne:>10d} {r:>8.2f}")

with open(f'{OUTPUT_DIR}/emergency_markers.txt', 'w', encoding='utf-8') as f:
    f.write("Топ-15 маркеров экстренности\n")
    f.write(f"{'Слово':20s} {'Экстр':>8s} {'Неэкстр':>10s} {'Ratio':>8s}\n")
    for word, e, ne, r in markers[:15]:
        f.write(f"{word:20s} {e:>8d} {ne:>10d} {r:>8.2f}\n")

