"""
Раздел 16.5: Тонкая настройка модели BERT на наборе данных IMDb
Использование DistilBERT из библиотеки transformers для классификации
обзоров фильмов на положительные/отрицательные.

Подразделы:
  16.5.1. Загрузка набора данных обзора фильмов IMDb
  16.5.2. Токенизация набора данных
  16.5.3. Загрузка и тонкая настройка предварительно обученной модели BERT
"""

import time
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import DistilBertTokenizerFast
from transformers import DistilBertForSequenceClassification


# ──────────────────────────────────────────────────────────────
# 16.5.1. Общие настройки и загрузка данных
# ──────────────────────────────────────────────────────────────

torch.backends.cudnn.deterministic = True
RANDOM_SEED = 123
torch.manual_seed(RANDOM_SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_EPOCHS = 3

print("=" * 70)
print("16.5. Тонкая настройка DistilBERT на IMDb")
print("=" * 70)
print(f"Устройство: {DEVICE}")
print(f"Количество эпох: {NUM_EPOCHS}")
print(f"Случайное начальное число: {RANDOM_SEED}")

# Загрузка данных (movie_data.csv из главы 8)
print("\nЗагрузка movie_data.csv...")
df = pd.read_csv('movie_data.csv')
print(f"Размер DataFrame: {df.shape}")
print(df.head(3))

# Разделение: 70% train, 10% valid, 20% test
train_texts = df.iloc[:35000]['review'].values
train_labels = df.iloc[:35000]['sentiment'].values
valid_texts = df.iloc[35000:40000]['review'].values
valid_labels = df.iloc[35000:40000]['sentiment'].values
test_texts = df.iloc[40000:]['review'].values
test_labels = df.iloc[40000:]['sentiment'].values

print(f"\nОбучение:   {len(train_texts)} примеров")
print(f"Валидация:  {len(valid_texts)} примеров")
print(f"Тестирование: {len(test_texts)} примеров")


# ──────────────────────────────────────────────────────────────
# 16.5.2. Токенизация набора данных
# ──────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("16.5.2. Токенизация с помощью DistilBertTokenizerFast")
print("=" * 70)

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

print("Токенизация обучающего набора...")
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True)
print("Токенизация валидационного набора...")
valid_encodings = tokenizer(list(valid_texts), truncation=True, padding=True)
print("Токенизация тестового набора...")
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True)
print("Токенизация завершена!")


class IMDbDataset(torch.utils.data.Dataset):
    """
    Пользовательский набор данных для IMDb на основе токенизированных кодировок.
    """
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

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=16, shuffle=True)
valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=16, shuffle=False)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=16, shuffle=False)

print(f"\nTrain DataLoader: {len(train_loader)} батчей")
print(f"Valid DataLoader: {len(valid_loader)} батчей")
print(f"Test  DataLoader: {len(test_loader)} батчей")


# ──────────────────────────────────────────────────────────────
# 16.5.3. Загрузка и тонкая настройка модели DistilBERT
# ──────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("16.5.3. Загрузка и тонкая настройка DistilBERT")
print("=" * 70)

model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased')
model.to(DEVICE)
model.train()
optim = torch.optim.Adam(model.parameters(), lr=5e-5)

print(model)


def compute_accuracy(model, data_loader, device):
    """
    Вычисление точности классификации на загруженном наборе данных.
    """
    with torch.no_grad():
        correct_pred, num_examples = 0, 0
        for batch_idx, batch in enumerate(data_loader):
            # Подготовка данных
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            predicted_labels = torch.argmax(logits, 1)

            num_examples += labels.size(0)
            correct_pred += (predicted_labels == labels).sum()

    return correct_pred.float() / num_examples * 100


# ── Цикл обучения (тонкая настройка) ──────────────────────────

print("\n" + "=" * 70)
print("Начало тонкой настройки...")
print("=" * 70)

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    model.train()

    for batch_idx, batch in enumerate(train_loader):
        # Подготовка данных
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)

        # Прямой проход
        outputs = model(input_ids,
                        attention_mask=attention_mask,
                        labels=labels)
        loss, logits = outputs['loss'], outputs['logits']

        # Обратный проход
        optim.zero_grad()
        loss.backward()
        optim.step()

        # Журналирование
        if not batch_idx % 250:
            print(f'Эпоха: {epoch + 1:04d}/{NUM_EPOCHS:04d} | '
                  f'Пакет {batch_idx:04d}/{len(train_loader):04d} | '
                  f'Потери: {loss:.4f}')

    # Оценка после эпохи
    model.eval()
    with torch.set_grad_enabled(False):
        train_acc = compute_accuracy(model, train_loader, DEVICE)
        valid_acc = compute_accuracy(model, valid_loader, DEVICE)
        print(f'Точность при обучении: {train_acc:.2f}%')
        print(f'Точность при валидации: {valid_acc:.2f}%')

    elapsed = (time.time() - start_time) / 60
    print(f'Времени прошло: {elapsed:.2f} min')

# Итоговая оценка на тестовом наборе
print("\n" + "=" * 70)
total_time = (time.time() - start_time) / 60
print(f'Общее время обучения: {total_time:.2f} min')

test_acc = compute_accuracy(model, test_loader, DEVICE)
print(f'Точность при тестировании: {test_acc:.2f}%')
print("=" * 70)
print("Готово!")
