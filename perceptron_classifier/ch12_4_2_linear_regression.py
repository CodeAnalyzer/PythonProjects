import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset, DataLoader

# ── Создание демонстрационного набора данных ──
X_train = np.arange(10, dtype='float32').reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0,
                    6.3, 6.6, 7.4, 8.0, 9.0], dtype='float32')

# Визуализация исходных данных
plt.figure(figsize=(6, 4))
plt.plot(X_train, y_train, 'o', markersize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Обучающие примеры')
plt.show()

# ── Стандартизация признаков ──
X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
X_train_norm = torch.from_numpy(X_train_norm)
y_train = torch.from_numpy(y_train).float()

# Создание Dataset и DataLoader
train_ds = TensorDataset(X_train_norm, y_train)
batch_size = 1
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# ── Определение модели линейной регрессии ──
torch.manual_seed(1)
weight = torch.randn(1)
weight.requires_grad_()
bias = torch.zeros(1, requires_grad=True)

def model(xb):
    return xb @ weight + bias

# ── Функция потерь (MSE) ──
def loss_fn(input, target):
    return (input - target).pow(2).mean()

# ── Обучение модели (SGD вручную) ──
learning_rate = 0.001
num_epochs = 200
log_epochs = 10

print('Обучение модели:')
for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        
        with torch.no_grad():
            weight -= weight.grad * learning_rate
            bias -= bias.grad * learning_rate
            weight.grad.zero_()
            bias.grad.zero_()
    
    if epoch % log_epochs == 0:
        print(f'Эпоха {epoch:3d}  Потеря {loss.item():.4f}')

# ── Вывод финальных параметров ──
print(f'\nОкончательные параметры: weight={weight.item():.4f}, bias={bias.item():.4f}')

# ── Визуализация результата ──
X_test = np.linspace(0, 9, num=100, dtype='float32').reshape(-1, 1)
X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)
X_test_norm = torch.from_numpy(X_test_norm)
y_pred = model(X_test_norm).detach().numpy()

fig, ax = plt.subplots(figsize=(8, 5))
plt.plot(X_train_norm.numpy(), y_train.numpy(), 'o', markersize=10, label='Обучающие примеры')
plt.plot(X_test_norm.numpy(), y_pred, '--', lw=3, label='Линейная регрессия')
plt.legend(fontsize=12)
ax.set_xlabel('x (стандартизированный)', size=12)
ax.set_ylabel('y', size=12)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.title('Результат линейной регрессии')
plt.tight_layout()
plt.show()
