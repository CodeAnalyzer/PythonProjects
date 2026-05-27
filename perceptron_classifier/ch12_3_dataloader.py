import torch
from torch.utils.data import DataLoader, TensorDataset

# ── 12.3.1. Создание DataLoader из существующего тензора ──

t = torch.arange(6, dtype=torch.float32)

print('Перебор элементов по одному:')
data_loader = DataLoader(t)
for item in data_loader:
    print(item)

print('\nПакетная обработка (batch_size=3):')
data_loader = DataLoader(t, batch_size=3, drop_last=False)
for i, batch in enumerate(data_loader, 1):
    print(f'batch {i}:', batch)

# ── 12.3.2. Объединение двух тензоров в совместный набор данных ──

torch.manual_seed(1)
t_x = torch.rand([4, 3], dtype=torch.float32)
t_y = torch.arange(4)

joint_dataset = TensorDataset(t_x, t_y)

print('\nПеребор совместного набора данных (признаки + метки):')
for example in joint_dataset:
    print('  x:', example[0], '  y:', example[1])

# ── 12.3.3. Перемешивание, группировка и повторение ──

print('\nПакеты с перемешиванием (batch_size=2, shuffle=True):')
torch.manual_seed(1)
data_loader = DataLoader(dataset=joint_dataset, batch_size=2, shuffle=True)
for i, batch in enumerate(data_loader, 1):
    print(f'batch {i}:  x:', batch[0], '\n         y:', batch[1])

print('\nОбучение в течение 2 эпох:')
for epoch in range(2):
    print(f'  epoch {epoch + 1}')
    for i, batch in enumerate(data_loader, 1):
        print(f'  batch {i}:  x:', batch[0], '\n           y:', batch[1])
