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

X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)
X_test_norm = torch.from_numpy(X_test_norm).float()
y_test_tensor = torch.from_numpy(y_test)

# Создание Dataset и DataLoader
train_ds = TensorDataset(X_train_norm, y_train)
torch.manual_seed(1)
batch_size = 2
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# ── Определение модели ──
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.layer1(x)
        x = nn.Sigmoid()(x)
        x = self.layer2(x)
        return x

input_size = X_train_norm.shape[1]
hidden_size = 16
output_size = 3

model = Model(input_size, hidden_size, output_size)

# ── Обучение модели ──
learning_rate = 0.001
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 100
print('Обучение модели:')
for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    if epoch % 20 == 0:
        print(f'  Эпоха {epoch}')

# Оценка точности перед сохранением
with torch.no_grad():
    pred_test = model(X_test_norm)
    correct = (torch.argmax(pred_test, dim=1) == y_test_tensor).float()
    accuracy_before = correct.mean()
    print(f'\nТочность на тесте (перед сохранением): {accuracy_before:.4f}')

# ── 12.4.6. Сохранение и загрузка модели ──

print('\n--- Сохранение всей модели ---')
path = 'iris_classifier.pt'
torch.save(model, path)
print(f'Модель сохранена в: {path}')

print('\n--- Загрузка всей модели ---')
model_new = torch.load(path, weights_only=False)
model_new.eval()
print('Загруженная модель:')
print(model_new)

# Проверка загруженной модели
with torch.no_grad():
    pred_test = model_new(X_test_norm)
    correct = (torch.argmax(pred_test, dim=1) == y_test_tensor).float()
    accuracy_loaded = correct.mean()
    print(f'Точность загруженной модели: {accuracy_loaded:.4f}')

print('\n--- Сохранение только параметров (state_dict) ---')
path_state = 'iris_classifier_state.pt'
torch.save(model.state_dict(), path_state)
print(f'Параметры сохранены в: {path_state}')

print('\n--- Загрузка параметров в новую модель ---')
model_new2 = Model(input_size, hidden_size, output_size)
model_new2.load_state_dict(torch.load(path_state))
model_new2.eval()
print('Параметры загружены в новую модель')

# Проверка
with torch.no_grad():
    pred_test = model_new2(X_test_norm)
    correct = (torch.argmax(pred_test, dim=1) == y_test_tensor).float()
    accuracy_state = correct.mean()
    print(f'Точность модели с загруженными параметрами: {accuracy_state:.4f}')

print(f'\n=== ИТОГО ===')
print(f'Исходная модель:  {accuracy_before:.4f}')
print(f'Полная загрузка:  {accuracy_loaded:.4f}')
print(f'State dict load:  {accuracy_state:.4f}')
