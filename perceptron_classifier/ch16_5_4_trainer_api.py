"""
Раздел 16.5.4: Удобная тонкая настройка трансформера с помощью API Trainer

Использование Trainer API из библиотеки transformers вместо ручного цикла обучения.
Повторно использует данные и токенизацию из 16.5.1–16.5.2.
"""

import time
import numpy as np
import pandas as pd
import torch
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)


# ──────────────────────────────────────────────────────────────
# Общие настройки
# ──────────────────────────────────────────────────────────────

torch.backends.cudnn.deterministic = True
RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 3

print("=" * 70)
print("16.5.4. Тонкая настройка через Trainer API")
print("=" * 70)
print(f"Устройство: {DEVICE}")


# ──────────────────────────────────────────────────────────────
# 16.5.1. Загрузка и разделение данных
# ──────────────────────────────────────────────────────────────

print("\nЗагрузка movie_data.csv...")
df = pd.read_csv('movie_data.csv')
print(f"Размер DataFrame: {df.shape}")

train_texts = df.iloc[:35000]['review'].values
train_labels = df.iloc[:35000]['sentiment'].values
valid_texts = df.iloc[35000:40000]['review'].values
valid_labels = df.iloc[35000:40000]['sentiment'].values
test_texts = df.iloc[40000:]['review'].values
test_labels = df.iloc[40000:]['sentiment'].values

print(f"Обучение: {len(train_texts)} | Валидация: {len(valid_texts)} | Тест: {len(test_texts)}")


# ──────────────────────────────────────────────────────────────
# 16.5.2. Токенизация
# ──────────────────────────────────────────────────────────────

print("\nТокенизация...")
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
valid_encodings = tokenizer(list(valid_texts), truncation=True, padding=True)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)
print("Токенизация завершена!")


class IMDbDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = IMDbDataset(train_encodings, train_labels)
valid_dataset = IMDbDataset(valid_encodings, valid_labels)
test_dataset = IMDbDataset(test_encodings, test_labels)


# ──────────────────────────────────────────────────────────────
# 16.5.4. Trainer API
# ──────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Загрузка модели DistilBERT...")
print("=" * 70)

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased')
model.to(DEVICE)
model.train()

optim = torch.optim.Adam(model.parameters(), lr=5e-5)

# Аргументы обучения
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy="epoch",
    save_strategy="epoch",
    report_to=[],
)

# Функция оценки точности
# В новой версии transformers используется библиотека evaluate
# вместо устаревшей datasets.load_metric
try:
    import evaluate
    metric = evaluate.load("accuracy")
except ImportError:
    print("Установка evaluate...")
    import subprocess
    subprocess.check_call(["pip", "install", "evaluate"])
    import evaluate
    metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# Trainer с валидационным набором для оценки после каждой эпохи
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    optimizers=(optim, None),
)

print("\n" + "=" * 70)
print("Начало обучения через Trainer API...")
print("=" * 70)

start_time = time.time()
trainer.train()
total_time = (time.time() - start_time) / 60
print(f"\nПолное время обучения: {total_time:.2f} min")


# ── Оценка на тестовом наборе ──────────────────────────────────

print("\n" + "=" * 70)
print("Оценка на тестовом наборе (trainer.evaluate)")
print("=" * 70)

# Оцениваем на тестовом наборе
test_trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir='./results',
        per_device_eval_batch_size=16,
        report_to=[],
    ),
    compute_metrics=compute_metrics,
)
test_results = test_trainer.evaluate(test_dataset)
print(f"\nРезультаты на тестовом наборе:")
for key, value in test_results.items():
    print(f"  {key}: {value}")


# ── Альтернативная оценка через compute_accuracy ───────────────

print("\n" + "=" * 70)
print("Альтернативная оценка через compute_accuracy")
print("=" * 70)


def compute_accuracy(model, data_loader, device):
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for batch_idx, batch in enumerate(data_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            predicted_labels = torch.argmax(logits, 1)

            num_examples += labels.size(0)
            correct_pred += (predicted_labels == labels).sum()

    return correct_pred.float() / num_examples * 100


test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=16, shuffle=False)

model.eval()
model.to(DEVICE)
test_acc = compute_accuracy(model, test_loader, DEVICE)
print(f"Точность при тестировании: {test_acc:.2f}%")

print("\n" + "=" * 70)
print("Готово!")
print("=" * 70)
