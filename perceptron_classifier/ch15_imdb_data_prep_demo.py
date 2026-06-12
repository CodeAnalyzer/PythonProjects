"""
Подготовка данных отзывов на фильмы (IMDB) для RNN
Глава 15: Предварительная обработка текстовых данных для классификации тональности
Использует реальные данные из movie_data.csv
"""

import re
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from collections import Counter, OrderedDict


class MovieDataset(Dataset):
    """
    Датасет отзывов на фильмы из movie_data.csv
    Колонки: review (текст), sentiment (0 или 1)
    """
    def __init__(self, csv_path):
        self.samples = []
        with open(csv_path, encoding='utf-8', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = float(row['sentiment'])
                text = row['review']
                self.samples.append((label, text))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def tokenizer(text):
    """
    Токенизация текста:
    - Удаление HTML-разметки
    - Сохранение эмодзи
    - Удаление знаков препинания и небуквенных символов
    - Приведение к нижнему регистру
    """
    # Удаление HTML-разметки
    text = re.sub('<[^>]*>', '', text)
    
    # Извлечение эмодзи
    emoticons = re.findall(
        r'(?:[:=;])(?:-)?(?:\)|\(|D|P)', text.lower()
    )
    
    # Удаление небуквенных символов и приведение к нижнему регистру
    text = re.sub(r'[\W]+', ' ', text.lower()) + \
        ' '.join(emoticons).replace('-', '')
    
    tokenized = text.split()
    return tokenized


# Глобальные пайплайны (инициализируются в main() после построения словаря)
text_pipeline = None
label_pipeline = None


def collate_batch(batch):
    """
    Функция для объединения примеров в батчи:
    - Кодирование текста в индексы
    - Преобразование меток
    - Padding последовательностей до одинаковой длины
    """
    label_list, text_list, lengths = [], [], []
    
    for _label, text in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(
            text_pipeline(text), dtype=torch.int64
        )
        text_list.append(processed_text)
        lengths.append(processed_text.size(0))
    
    label_list = torch.tensor(label_list)
    lengths = torch.tensor(lengths)
    
    # Padding последовательностей
    padded_text_list = nn.utils.rnn.pad_sequence(
        text_list, batch_first=True
    )
    
    return padded_text_list, label_list, lengths


def main():
    print("=" * 70)
    print("Подготовка данных отзывов на фильмы (IMDB) для RNN")
    print("(Данные из movie_data.csv)")
    print("=" * 70)
    
    # Шаг 1: Загрузка данных из CSV
    print("\nШаг 1: Загрузка данных из movie_data.csv")
    print("-" * 70)
    
    csv_path = 'movie_data.csv'
    full_dataset = MovieDataset(csv_path)
    
    print(f"Всего примеров: {len(full_dataset)}")
    
    # Подсчёт классов
    pos = sum(1 for label, _ in full_dataset.samples if label == 1.0)
    neg = len(full_dataset) - pos
    print(f"Позитивных (1): {pos}")
    print(f"Негативных (0): {neg}")
    
    # Пример данных
    print("\nПример из датасета:")
    label, text = full_dataset[0]
    print(f"Метка: {label} ({'pos' if label == 1.0 else 'neg'})")
    print(f"Текст (первые 200 символов): {text[:200]}...")
    
    # Разделение на train / valid / test
    print("\nРазделение на train/valid/test...")
    torch.manual_seed(1)
    total = len(full_dataset)
    test_size = 5000
    valid_size = 5000
    train_size = total - valid_size - test_size
    train_dataset, valid_dataset, test_dataset = random_split(
        full_dataset, (train_size, valid_size, test_size)
    )
    
    print(f"Размер обучающего набора: {len(train_dataset)}")
    print(f"Размер валидационного набора: {len(valid_dataset)}")
    print(f"Размер тестового набора: {len(test_dataset)}")
    
    # Шаг 2: Поиск уникальных токенов
    print("\n" + "=" * 70)
    print("Шаг 2: Поиск уникальных токенов (слов)")
    print("-" * 70)
    
    token_counts = Counter()
    
    print("Сбор токенов из обучающего набора...")
    for label, line in train_dataset:
        tokens = tokenizer(line)
        token_counts.update(tokens)
    
    print(f"Размер словаря: {len(token_counts)}")
    print(f"Топ-10 самых частых слов:")
    for token, count in token_counts.most_common(10):
        print(f"  '{token}': {count}")
    
    # Шаг 3: Кодирование токенов целыми числами
    print("\n" + "=" * 70)
    print("Шаг 3: Создание словаря и кодирование токенов")
    print("-" * 70)
    
    # Сортировка по частоте
    sorted_by_freq_tuples = sorted(
        token_counts.items(), key=lambda x: x[1], reverse=True
    )
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    
    # Создание словаря вручную (как в torchtext.vocab)
    word_to_idx = {}
    idx_to_word = {}
    
    # Добавляем специальные токены
    word_to_idx['<pad>'] = 0
    word_to_idx['<unk>'] = 1
    idx_to_word[0] = '<pad>'
    idx_to_word[1] = '<unk>'
    
    # Добавляем слова из словаря
    for idx, (word, _) in enumerate(ordered_dict.items(), start=2):
        word_to_idx[word] = idx
        idx_to_word[idx] = word
    
    print(f"Размер словаря с <pad> и <unk>: {len(word_to_idx)}")
    print(f"Индекс <pad>: {word_to_idx['<pad>']}")
    print(f"Индекс <unk>: {word_to_idx['<unk>']}")
    
    # Пример кодирования
    example_tokens = ['this', 'is', 'an', 'example']
    example_indices = [word_to_idx.get(token, word_to_idx['<unk>']) 
                       for token in example_tokens]
    print(f"\nПример кодирования:")
    print(f"  Токены: {example_tokens}")
    print(f"  Индексы: {example_indices}")
    
    # Определение пайплайнов (глобальные, чтобы collate_batch мог их использовать)
    global text_pipeline, label_pipeline
    text_pipeline = lambda x: [word_to_idx.get(token, word_to_idx['<unk>']) 
                               for token in tokenizer(x)]
    label_pipeline = lambda x: 1.0 if x == 1.0 else 0.0
    
    # Шаг 4: Демонстрация padding
    print("\n" + "=" * 70)
    print("Шаг 4: Демонстрация padding для батчей")
    print("-" * 70)
    
    # Создание небольшого загрузчика для демонстрации
    demo_dataloader = DataLoader(
        train_dataset, batch_size=4,
        shuffle=False, collate_fn=collate_batch
    )
    
    text_batch, label_batch, length_batch = next(iter(demo_dataloader))
    
    print(f"Размер батча текстов: {text_batch.shape}")
    print(f"Размер батча меток: {label_batch.shape}")
    print(f"Длины последовательностей: {length_batch.tolist()}")
    print(f"Максимальная длина: {length_batch.max().item()}")
    
    print("\nПервый пример в батче (первые 20 токенов):")
    print(f"  Текст: {text_batch[0, :20].tolist()}")
    print(f"  Метка: {label_batch[0].item()}")
    print(f"  Длина: {length_batch[0].item()}")
    
    # Шаг 5: Создание финальных загрузчиков данных
    print("\n" + "=" * 70)
    print("Шаг 5: Создание финальных DataLoader'ов")
    print("-" * 70)
    
    batch_size = 32
    
    train_dl = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, collate_fn=collate_batch
    )
    valid_dl = DataLoader(
        valid_dataset, batch_size=batch_size,
        shuffle=False, collate_fn=collate_batch
    )
    test_dl = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, collate_fn=collate_batch
    )
    
    print(f"Размер батча: {batch_size}")
    print(f"Количество батчей в train: {len(train_dl)}")
    print(f"Количество батчей в valid: {len(valid_dl)}")
    print(f"Количество батчей в test: {len(test_dl)}")
    
    # Проверка батча
    text_batch, label_batch, length_batch = next(iter(train_dl))
    print(f"\nПример батча из train_dl:")
    print(f"  Форма текстов: {text_batch.shape}")
    print(f"  Форма меток: {label_batch.shape}")
    print(f"  Форма длин: {length_batch.shape}")
    
    print("\n" + "=" * 70)
    print("Подготовка данных завершена!")
    print("=" * 70)
    print(f"""
Итоги:
1. Загружено {total} отзывов из movie_data.csv
2. Разделение: {train_size} train / {valid_size} valid / {test_size} test
3. Создан словарь из уникальных слов
4. Добавлены специальные токены <pad> (0) и <unk> (1)
5. Созданы text_pipeline и label_pipeline
6. Реализован padding последовательностей
7. Созданы DataLoader'ы с batch_size=32

Данные готовы для обучения RNN модели!
    """)


if __name__ == '__main__':
    main()
