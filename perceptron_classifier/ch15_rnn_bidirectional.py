"""
Раздел 15.4: Двунаправленная RNN для анализа тональности
Bidirectional LSTM: проходит последовательность в обоих направлениях
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
    Токенизация текста
    """
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall(
        r'(?:[:=;])(?:-)?(?:\)|\(|D|P)', text.lower()
    )
    text = re.sub(r'[\W]+', ' ', text.lower()) + \
        ' '.join(emoticons).replace('-', '')
    tokenized = text.split()
    return tokenized


# Глобальные пайплайны
text_pipeline = None
label_pipeline = None


def collate_batch(batch):
    """
    Функция для объединения примеров в батчи
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

    padded_text_list = nn.utils.rnn.pad_sequence(
        text_list, batch_first=True
    )

    return padded_text_list, label_list, lengths


class BiRNNModel(nn.Module):
    """
    Двунаправленная модель RNN (Bidirectional LSTM)
    Embedding -> BiLSTM -> FC1 -> ReLU -> FC2 -> Sigmoid
    """
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size):
        super().__init__()

        # Embedding слой
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # Двунаправленный LSTM слой
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size,
                           batch_first=True, bidirectional=True)

        # Полносвязные слои
        # Вход * 2, так как bidirectional объединяет прямое и обратное скрытые состояния
        self.fc1 = nn.Linear(rnn_hidden_size * 2, fc_hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(fc_hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text, lengths):
        """
        Прямой проход
        text: (batch_size, seq_len) - индексы слов
        lengths: (batch_size) - реальные длины последовательностей
        """
        # Embedding: (batch_size, seq_len, embed_dim)
        out = self.embedding(text)

        # Pack padded sequence для эффективной обработки разной длины
        out = nn.utils.rnn.pack_padded_sequence(
            out, lengths.cpu().numpy(), enforce_sorted=False, batch_first=True
        )

        # LSTM: output, (hidden, cell)
        out, (hidden, cell) = self.rnn(out)

        # Bidirectional: hidden имеет форму (2, batch_size, hidden_size)
        # hidden[-2, :, :] - последнее скрытое состояние обратного прохода
        # hidden[-1, :, :] - последнее скрытое состояние прямого прохода
        # Конкатенируем их по dim=1
        out = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)

        # FC1 + ReLU
        out = self.fc1(out)
        out = self.relu(out)

        # FC2 + Sigmoid
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out


def train(dataloader, model, loss_fn, optimizer, device):
    """
    Обучение модели на одной эпохе
    """
    model.train()
    total_acc, total_loss = 0, 0

    for text_batch, label_batch, lengths in dataloader:
        text_batch = text_batch.to(device)
        label_batch = label_batch.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        pred = model(text_batch, lengths)[:, 0]
        loss = loss_fn(pred, label_batch)
        loss.backward()
        optimizer.step()

        total_acc += ((pred >= 0.5).float() == label_batch).float().sum().item()
        total_loss += loss.item() * label_batch.size(0)

    return total_acc / len(dataloader.dataset), total_loss / len(dataloader.dataset)


def evaluate(dataloader, model, loss_fn, device):
    """
    Оценка модели на датасете
    """
    model.eval()
    total_acc, total_loss = 0, 0

    with torch.no_grad():
        for text_batch, label_batch, lengths in dataloader:
            text_batch = text_batch.to(device)
            label_batch = label_batch.to(device)
            lengths = lengths.to(device)

            pred = model(text_batch, lengths)[:, 0]
            loss = loss_fn(pred, label_batch)

            total_acc += ((pred >= 0.5).float() == label_batch).float().sum().item()
            total_loss += loss.item() * label_batch.size(0)

    return total_acc / len(dataloader.dataset), total_loss / len(dataloader.dataset)


def prepare_data():
    """
    Подготовка данных: загрузка, словарь, DataLoader'ы
    """
    print("Загрузка данных из movie_data.csv...")
    full_dataset = MovieDataset('movie_data.csv')
    print(f"Всего примеров: {len(full_dataset)}")

    # Разделение
    torch.manual_seed(1)
    total = len(full_dataset)
    test_size = 5000
    valid_size = 5000
    train_size = total - valid_size - test_size
    train_dataset, valid_dataset, test_dataset = random_split(
        full_dataset, (train_size, valid_size, test_size)
    )

    print(f"Train: {train_size}, Valid: {valid_size}, Test: {test_size}")

    # Построение словаря
    print("\nПостроение словаря...")
    token_counts = Counter()
    for label, line in train_dataset:
        tokens = tokenizer(line)
        token_counts.update(tokens)

    print(f"Размер словаря: {len(token_counts)}")

    # Создание словаря
    sorted_by_freq_tuples = sorted(
        token_counts.items(), key=lambda x: x[1], reverse=True
    )
    ordered_dict = OrderedDict(sorted_by_freq_tuples)

    word_to_idx = {}
    word_to_idx['<pad>'] = 0
    word_to_idx['<unk>'] = 1
    for idx, (word, _) in enumerate(ordered_dict.items(), start=2):
        word_to_idx[word] = idx

    vocab_size = len(word_to_idx)
    print(f"Размер словаря с <pad> и <unk>: {vocab_size}")

    # Глобальные пайплайны
    global text_pipeline, label_pipeline
    text_pipeline = lambda x: [word_to_idx.get(token, word_to_idx['<unk>'])
                               for token in tokenizer(x)]
    label_pipeline = lambda x: 1.0 if x == 1.0 else 0.0

    # DataLoader'ы
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

    return train_dl, valid_dl, test_dl, vocab_size


def main():
    print("=" * 70)
    print("15.4. Двунаправленная RNN (Bidirectional LSTM)")
    print("=" * 70)

    # Подготовка данных
    train_dl, valid_dl, test_dl, vocab_size = prepare_data()

    # Параметры модели
    embed_dim = 20
    rnn_hidden_size = 64
    fc_hidden_size = 64

    print("\n" + "=" * 70)
    print("Создание модели Bidirectional LSTM")
    print("=" * 70)

    torch.manual_seed(1)
    model = BiRNNModel(vocab_size, embed_dim, rnn_hidden_size, fc_hidden_size)
    print(model)

    # Функция потерь и оптимизатор
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Определение устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"\nИспользуемое устройство: {device}")

    # Обучение
    print("\n" + "=" * 70)
    print("Обучение модели")
    print("=" * 70)

    num_epochs = 10

    for epoch in range(num_epochs):
        acc_train, loss_train = train(train_dl, model, loss_fn, optimizer, device)
        acc_valid, loss_valid = evaluate(valid_dl, model, loss_fn, device)

        print(f'Эпоха {epoch}: '
              f'train_acc: {acc_train:.4f} train_loss: {loss_train:.4f} | '
              f'val_acc: {acc_valid:.4f} val_loss: {loss_valid:.4f}')

    # Оценка на тестовом наборе
    print("\n" + "=" * 70)
    print("Оценка на тестовом наборе")
    print("=" * 70)

    acc_test, loss_test = evaluate(test_dl, model, loss_fn, device)
    print(f'test_acc: {acc_test:.4f} test_loss: {loss_test:.4f}')

    print("\n" + "=" * 70)
    print("Обучение завершено!")
    print("=" * 70)
    print("""
Двунаправленный слой RNN выполняет два прохода:
- Прямой проход (от начала до конца)
- Обратный проход (от конца до начала)

Результирующие скрытые состояния объединяются конкатенацией:
- hidden[-2, :, :] - финальное скрытое состояние обратного прохода
- hidden[-1, :, :] - финальное скрытое состояние прямого прохода
- torch.cat(..., dim=1) -> (batch_size, hidden_size * 2)

Это позволяет модели учитывать контекст из обоих концов последовательности.
    """)


if __name__ == '__main__':
    main()
