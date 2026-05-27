import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ── Создание демонстрационного набора данных (тот же, что в 12.4.2) ──
X_train = np.arange(10, dtype='float32').reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0,
                    6.3, 6.6, 7.4, 8.0, 9.0], dtype='float32')

# Стандартизация признаков
X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
X_train_norm = torch.from_numpy(X_train_norm)
y_train = torch.from_numpy(y_train).float()

# Создание Dataset и DataLoader
train_ds = TensorDataset(X_train_norm, y_train)
batch_size = 1
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# ── Определение модели с помощью torch.nn ──
input_size = 1
output_size = 1
model = nn.Linear(input_size, output_size)

# ── Функция потерь и оптимизатор ──
loss_fn = nn.MSELoss(reduction='mean')
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# ── Обучение модели ──
num_epochs = 200
log_epochs = 10

print('Обучение модели с помощью torch.nn и torch.optim:')
for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        # 1. Генерируем прогнозы
        pred = model(x_batch)[:, 0]
        
        # 2. Вычисляем потери
        loss = loss_fn(pred, y_batch)
        
        # 3. Вычисляем градиенты
        loss.backward()
        
        # 4. Обновляем параметры, используя градиенты
        optimizer.step()
        
        # 5. Обнуляем градиенты
        optimizer.zero_grad()
    
    if epoch % log_epochs == 0:
        print(f'Эпоха {epoch:3d}  Потери {loss.item():.4f}')

# ── Вывод финальных параметров ──
print(f'\nОкончательные параметры: weight={model.weight.item():.4f}, bias={model.bias.item():.4f}')

# ── Визуализация результата ──
X_test = np.linspace(0, 9, num=100, dtype='float32').reshape(-1, 1)
X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)
X_test_norm = torch.from_numpy(X_test_norm)
y_pred = model(X_test_norm).detach().numpy()

fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(X_train_norm.numpy(), y_train.numpy(), 'o', markersize=10, label='Обучающие примеры')
plt.plot(X_test_norm.numpy(), y_pred, '--', lw=3, label='Линейная регрессия (nn.Linear)')
plt.legend(fontsize=12)
ax.set_xlabel('x (стандартизированный)', size=12)
ax.set_ylabel('y', size=12)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.title('Результат линейной регрессии (torch.nn + torch.optim)')
plt.tight_layout()
plt.show()
