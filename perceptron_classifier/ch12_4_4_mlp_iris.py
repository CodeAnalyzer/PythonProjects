import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# ── Загрузка и подготовка данных ──
iris = load_iris()
X = iris['data']
y = iris['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1./3, random_state=1)

# Стандартизация признаков
X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
X_train_norm = torch.from_numpy(X_train_norm).float()
y_train = torch.from_numpy(y_train)

# Создание Dataset и DataLoader
train_ds = TensorDataset(X_train_norm, y_train)
torch.manual_seed(1)
batch_size = 2
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# ── Определение модели (двухслойный персептрон) ──
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = nn.Sigmoid()(x)
        x = self.layer2(x)
        # Softmax не нужен - CrossEntropyLoss уже включает его
        return x

input_size = X_train_norm.shape[1]
hidden_size = 16
output_size = 3

model = Model(input_size, hidden_size, output_size)

# ── Функция потерь и оптимизатор ──
learning_rate = 0.001
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ── Обучение модели ──
num_epochs = 100
loss_hist = [0] * num_epochs
accuracy_hist = [0] * num_epochs

print('Обучение модели классификации ирисов:')
for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        loss_hist[epoch] += loss.item() * y_batch.size(0)
        is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
        accuracy_hist[epoch] += is_correct.sum()
    
    loss_hist[epoch] /= len(train_dl.dataset)
    accuracy_hist[epoch] /= len(train_dl.dataset)
    
    if epoch % 10 == 0:
        print(f'Эпоха {epoch:3d}  Потеря: {loss_hist[epoch]:.4f}  Точность: {accuracy_hist[epoch]:.4f}')

print(f'\nФинальная точность на обучении: {accuracy_hist[-1]:.4f}')

# ── Визуализация кривых обучения ──
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(loss_hist, lw=3)
axes[0].set_title('Training loss', size=15)
axes[0].set_xlabel('Epoch', size=15)
axes[0].set_ylabel('Loss', size=15)
axes[0].tick_params(axis='both', which='major', labelsize=12)

axes[1].plot(accuracy_hist, lw=3)
axes[1].set_title('Training accuracy', size=15)
axes[1].set_xlabel('Epoch', size=15)
axes[1].set_ylabel('Accuracy', size=15)
axes[1].tick_params(axis='both', which='major', labelsize=12)
axes[1].set_ylim([0, 1.05])

plt.tight_layout()
plt.show()

# ── Оценка на тестовых данных ──
X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)
X_test_norm = torch.from_numpy(X_test_norm).float()
y_test_tensor = torch.from_numpy(y_test)

with torch.no_grad():
    test_pred = model(X_test_norm)
    test_accuracy = (torch.argmax(test_pred, dim=1) == y_test_tensor).float().mean()
    print(f'Точность на тесте: {test_accuracy:.4f}')
