"""
Раздел 15.3.2: Моделирование языка на уровне символов в PyTorch
Генерация текста по стилю входного документа с помощью RNN
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def load_and_preprocess_text(filepath):
    """
    Загрузка и предварительная обработка текста
    """
    with open(filepath, 'r', encoding='utf-8') as fp:
        text = fp.read()

    # Обрезаем начало и конец (метаданные Project Gutenberg)
    start_indx = text.find('THE MYSTERIOUS ISLAND')
    end_indx = text.find('End of the Project Gutenberg')
    text = text[start_indx:end_indx]

    # Уникальные символы
    char_set = set(text)
    chars_sorted = sorted(char_set)

    # Словари для кодирования
    char2int = {ch: i for i, ch in enumerate(chars_sorted)}
    char_array = np.array(chars_sorted)

    # Кодирование текста
    text_encoded = np.array(
        [char2int[ch] for ch in text],
        dtype=np.int32
    )

    print(f'Общая длина текста: {len(text)}')
    print(f'Уникальных символов: {len(char_set)}')

    return text, text_encoded, char_array, char2int


class TextDataset(Dataset):
    """
    Датасет для моделирования языка на уровне символов
    """
    def __init__(self, text_chunks):
        self.text_chunks = text_chunks

    def __len__(self):
        return len(self.text_chunks)

    def __getitem__(self, idx):
        text_chunk = self.text_chunks[idx]
        return text_chunk[:-1].long(), text_chunk[1:].long()


def prepare_data(text_encoded, seq_length=40):
    """
    Создание чанков и DataLoader
    """
    chunk_size = seq_length + 1
    text_chunks = [
        text_encoded[i:i+chunk_size]
        for i in range(len(text_encoded) - chunk_size + 1)
    ]

    dataset = TextDataset(torch.tensor(np.array(text_chunks)))

    batch_size = 64
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=True, drop_last=True
    )

    print(f'Количество фрагментов: {len(dataset)}')
    print(f'Размер батча: {batch_size}')
    print(f'Длина последовательности: {seq_length}')

    return dataloader


class CharRNN(nn.Module):
    """
    Модель RNN для генерации текста на уровне символов
    Обрабатывает по одному символу за раз
    """
    def __init__(self, vocab_size, embed_dim, rnn_hidden_size):
        super().__init__()
        self.rnn_hidden_size = rnn_hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, rnn_hidden_size, batch_first=True)
        self.fc = nn.Linear(rnn_hidden_size, vocab_size)

    def forward(self, x, hidden, cell):
        """
        Прямой проход для одного символа
        x: (batch_size,) - индекс символа
        hidden: (1, batch_size, hidden_size)
        cell: (1, batch_size, hidden_size)
        """
        # Embedding: (batch_size, 1, embed_dim)
        out = self.embedding(x).unsqueeze(1)
        # LSTM: (batch_size, 1, hidden_size)
        out, (hidden, cell) = self.rnn(out, (hidden, cell))
        # FC: (batch_size, vocab_size)
        out = self.fc(out).reshape(out.size(0), -1)
        return out, hidden, cell

    def init_hidden(self, batch_size, device):
        """
        Инициализация скрытого состояния
        """
        hidden = torch.zeros(1, batch_size, self.rnn_hidden_size, device=device)
        cell = torch.zeros(1, batch_size, self.rnn_hidden_size, device=device)
        return hidden, cell


from torch.distributions.categorical import Categorical


def generate_text(model, start_str, char_array, char2int,
                  len_generated_text=500, scale_factor=1.0, device='cpu'):
    """
    Генерация текста из обученной модели с использованием Categorical
    и масштабированием логитов (температура).
    """
    model.eval()

    # Кодируем начальную строку
    encoded_input = torch.tensor(
        [char2int[s] for s in start_str], device=device
    )
    encoded_input = torch.reshape(encoded_input, (1, -1))

    generated_str = start_str

    hidden, cell = model.init_hidden(1, device)

    with torch.no_grad():
        # Пропускаем начальную строку через модель
        for c in range(len(start_str) - 1):
            _, hidden, cell = model(
                encoded_input[:, c].view(1), hidden, cell
            )

        last_char = encoded_input[:, -1]

        # Генерируем новые символы
        for _ in range(len_generated_text):
            logits, hidden, cell = model(
                last_char.view(1), hidden, cell
            )
            logits = torch.squeeze(logits, 0)
            scaled_logits = logits * scale_factor
            m = Categorical(logits=scaled_logits)
            last_char = m.sample()
            generated_str += str(char_array[last_char])

    return generated_str


def main():
    print("=" * 70)
    print("15.3.2. Моделирование языка на уровне символов")
    print("=" * 70)

    # Загрузка и подготовка данных
    print("\nЗагрузка текста...")
    text, text_encoded, char_array, char2int = load_and_preprocess_text('1268-0.txt')

    # Примеры кодирования/декодирования
    print(f"\nПервые 15 символов текста:")
    print(f"  Текст: {repr(text[:15])}")
    print(f"  Код: {text_encoded[:15]}")
    print(f"\nСимволы 15-20:")
    print(f"  Код: {text_encoded[15:21]}")
    print(f"  Декод: {repr(''.join(char_array[text_encoded[15:21]]))}")

    # Сопоставления первых 5 символов
    print("\nСопоставления символ -> код:")
    for ex in text_encoded[:5]:
        print(f'  {ex} -> {char_array[ex]}')

    # Подготовка данных
    print("\n" + "-" * 70)
    seq_length = 40
    dataloader = prepare_data(text_encoded, seq_length)

    # Демонстрация вход/цель
    print("\nПример вход/целевой последовательности:")
    for i, (seq, target) in enumerate(dataloader.dataset):
        print(f"  Вход (x): {repr(''.join(char_array[seq]))}")
        print(f"  Цель (y): {repr(''.join(char_array[target]))}")
        if i == 0:
            break

    # Создание модели
    print("\n" + "=" * 70)
    print("Создание модели CharRNN")
    print("=" * 70)

    vocab_size = len(char_array)
    embed_dim = 256
    rnn_hidden_size = 512

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    torch.manual_seed(1)
    model = CharRNN(vocab_size, embed_dim, rnn_hidden_size)
    model = model.to(device)
    print(model)
    print(f"\nvocab_size: {vocab_size}, embed_dim: {embed_dim}, hidden_size: {rnn_hidden_size}")

    # Проверка сохранённой модели
    import os
    model_path = 'char_rnn_model.pth'
    vocab_path = 'char_rnn_vocab.npz'

    if os.path.exists(model_path) and os.path.exists(vocab_path):
        print("\nНайдена сохранённая модель, загрузка...")
        model.load_state_dict(torch.load(model_path, map_location=device))
        vocab_data = np.load(vocab_path)
        char_array = vocab_data['char_array']
        char2int = dict(zip(vocab_data['char2int_keys'], vocab_data['char2int_values'].tolist()))
        print("Модель загружена!")
        skip_training = True
    else:
        print("\nСохранённая модель не найдена, начинаем обучение...")
        skip_training = False

    # Функция потерь и оптимизатор
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    # Обучение
    print("\n" + "=" * 70)
    print("Обучение модели")
    print("=" * 70)

    num_epochs = 10000
    batch_size = 64

    if not skip_training:
        torch.manual_seed(1)

        for epoch in range(num_epochs):
            hidden, cell = model.init_hidden(batch_size, device)
            seq_batch, target_batch = next(iter(dataloader))
            seq_batch = seq_batch.to(device)
            target_batch = target_batch.to(device)

            optimizer.zero_grad()
            loss = 0

            for c in range(seq_length):
                pred, hidden, cell = model(seq_batch[:, c], hidden, cell)
                loss += loss_fn(pred, target_batch[:, c])

            loss.backward()
            optimizer.step()

            loss = loss.item() / seq_length

            if epoch % 500 == 0:
                print(f'Эпоха {epoch} потери: {loss:.4f}')

        print(f'Эпоха {num_epochs} потери: {loss:.4f}')

        # Сохранение модели
        torch.save(model.state_dict(), 'char_rnn_model.pth')
        np.savez('char_rnn_vocab.npz',
                 char_array=char_array,
                 char2int_keys=list(char2int.keys()),
                 char2int_values=list(char2int.values()))
        print("Модель сохранена в 'char_rnn_model.pth'")
        print("Словарь сохранён в 'char_rnn_vocab.npz'")

    # Генерация текста
    print("\n" + "=" * 70)
    print("Генерация текста")
    print("=" * 70)

    start_str = "The island"

    # scale_factor=1.0 (по умолчанию)
    torch.manual_seed(1)
    generated_text = generate_text(
        model, start_str, char_array, char2int,
        len_generated_text=500, scale_factor=1.0, device=device
    )
    print(f"\nНачальная строка: '{start_str}'")
    print(f"scale_factor=1.0:")
    print(generated_text)

    # scale_factor=2.0 (более предсказуемо)
    torch.manual_seed(1)
    generated_text = generate_text(
        model, start_str, char_array, char2int,
        len_generated_text=500, scale_factor=2.0, device=device
    )
    print(f"\nscale_factor=2.0 (более предсказуемо):")
    print(generated_text)

    # scale_factor=0.5 (более случайно)
    torch.manual_seed(1)
    generated_text = generate_text(
        model, start_str, char_array, char2int,
        len_generated_text=500, scale_factor=0.5, device=device
    )
    print(f"\nscale_factor=0.5 (более случайно):")
    print(generated_text)

    print("\n" + "=" * 70)
    print("Готово!")
    print("=" * 70)


if __name__ == '__main__':
    main()
