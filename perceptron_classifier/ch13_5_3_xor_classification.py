import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── Генерация данных XOR ──
torch.manual_seed(1)
np.random.seed(1)

x = np.random.uniform(low=-1, high=1, size=(200, 2))
y = np.ones(len(x))
y[x[:, 0] * x[:, 1] < 0] = 0

n_train = 100
x_train = torch.tensor(x[:n_train, :], dtype=torch.float32)
y_train = torch.tensor(y[:n_train], dtype=torch.float32)
x_valid = torch.tensor(x[n_train:, :], dtype=torch.float32)
y_valid = torch.tensor(y[n_train:], dtype=torch.float32)

# ── Визуализация данных ──
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(x[y == 0, 0], x[y == 0, 1], 'o', alpha=0.75, markersize=10, label='Класс 0')
ax.plot(x[y == 1, 0], x[y == 1, 1], '<', alpha=0.75, markersize=10, label='Класс 1')
ax.set_xlabel(r'$x_1$', size=15)
ax.set_ylabel(r'$x_2$', size=15)
ax.legend(fontsize=12)
ax.set_title('Данные XOR', size=15)
plt.tight_layout()
plt.show()

# ── Функция обучения ──
def train(model, num_epochs, train_dl, x_valid, y_valid, n_train, batch_size):
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    
    for epoch in range(num_epochs):
        for x_batch, y_batch in train_dl:
            pred = model(x_batch)[:, 0]
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            loss_hist_train[epoch] += loss.item()
            is_correct = ((pred > 0.5).float() == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.mean()
        
        loss_hist_train[epoch] /= n_train / batch_size
        accuracy_hist_train[epoch] /= n_train / batch_size
        
        pred = model(x_valid)[:, 0]
        loss = loss_fn(pred, y_valid)
        loss_hist_valid[epoch] = loss.item()
        is_correct = ((pred > 0.5).float() == y_valid).float()
        accuracy_hist_valid[epoch] += is_correct.mean()
    
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid

# ── МОДЕЛЬ 1: Простая логистическая регрессия (без скрытых слоёв) ──
print('=== МОДЕЛЬ 1: Логистическая регрессия ===')
model1 = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)
print(model1)

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model1.parameters(), lr=0.001)

train_ds = TensorDataset(x_train, y_train)
batch_size = 2
torch.manual_seed(1)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

num_epochs = 200
history1 = train(model1, num_epochs, train_dl, x_valid, y_valid, n_train, batch_size)

print(f'Финальная точность (обучение): {history1[2][-1]:.4f}')
print(f'Финальная точность (валидация): {history1[3][-1]:.4f}')

# Визуализация результатов модели 1
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
axes[0].plot(history1[0], lw=4, label='Потери при обучении')
axes[0].plot(history1[1], lw=4, label='Потери при валидации')
axes[0].legend(fontsize=12)
axes[0].set_xlabel('Эпохи', size=12)
axes[0].set_ylabel('Потеря', size=12)
axes[0].set_title('Модель 1: Логистическая регрессия - Потери', size=14)

axes[1].plot(history1[2], lw=4, label='Точность при обучении')
axes[1].plot(history1[3], lw=4, label='Точность при валидации')
axes[1].legend(fontsize=12)
axes[1].set_xlabel('Эпохи', size=12)
axes[1].set_ylabel('Точность', size=12)
axes[1].set_title('Модель 1: Логистическая регрессия - Точность', size=12)
plt.tight_layout()
plt.show()

# ── МОДЕЛЬ 2: MLP с двумя скрытыми слоями ──
print('\n=== МОДЕЛЬ 2: MLP с двумя скрытыми слоями ===')
model2 = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
    nn.Sigmoid()
)
print(model2)

loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model2.parameters(), lr=0.015)

history2 = train(model2, num_epochs, train_dl, x_valid, y_valid, n_train, batch_size)

print(f'Финальная точность (обучение): {history2[2][-1]:.4f}')
print(f'Финальная точность (валидация): {history2[3][-1]:.4f}')

# Визуализация результатов модели 2
fig, axes = plt.subplots(1, 2, figsize=(16, 4))
axes[0].plot(history2[0], lw=4, label='Потери при обучении')
axes[0].plot(history2[1], lw=4, label='Потери при валидации')
axes[0].legend(fontsize=12)
axes[0].set_xlabel('Эпохи', size=12)
axes[0].set_ylabel('Потеря', size=12)
axes[0].set_title('Модель 2: MLP (2 скрытых слоя) - Потери', size=14)

axes[1].plot(history2[2], lw=4, label='Точность при обучении')
axes[1].plot(history2[3], lw=4, label='Точность при валидации')
axes[1].legend(fontsize=12)
axes[1].set_xlabel('Эпохи', size=12)
axes[1].set_ylabel('Точность', size=12)
axes[1].set_title('Модель 2: MLP (2 скрытых слоя) - Точность', size=12)
plt.tight_layout()
plt.show()

# ── Сравнение результатов ──
print('\n=== СРАВНЕНИЕ МОДЕЛЕЙ ===')
print(f'Модель 1 (логистическая регрессия):')
print(f'  Точность на обучении: {history1[2][-1]:.4f}')
print(f'  Точность на валидации: {history1[3][-1]:.4f}')
print(f'\nМодель 2 (MLP с 2 скрытыми слоями):')
print(f'  Точность на обучении: {history2[2][-1]:.4f}')
print(f'  Точность на валидации: {history2[3][-1]:.4f}')
