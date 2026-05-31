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

# ── Пользовательский слой NoisyLinear ──
class NoisyLinear(nn.Module):
    def __init__(self, input_size, output_size, noise_stddev=0.1):
        super().__init__()
        w = torch.Tensor(input_size, output_size)
        self.w = nn.Parameter(w)
        nn.init.xavier_uniform_(self.w)
        b = torch.Tensor(output_size).fill_(0)
        self.b = nn.Parameter(b)
        self.noise_stddev = noise_stddev
    
    def forward(self, x, training=False):
        if training:
            noise = torch.normal(0.0, self.noise_stddev, x.shape)
            x_new = torch.add(x, noise)
        else:
            x_new = x
        return torch.add(torch.mm(x_new, self.w), self.b)

# ── Тестирование слоя NoisyLinear ──
print('Тестирование слоя NoisyLinear:')
torch.manual_seed(1)
noisy_layer = NoisyLinear(4, 2)
x_test = torch.zeros((1, 4))

print('  С шумом (training=True):')
print('    ', noisy_layer(x_test, training=True))
print('    ', noisy_layer(x_test, training=True))
print('  Без шума (training=False):')
print('    ', noisy_layer(x_test, training=False))

# ── Модель с пользовательским слоем ──
class MyNoisyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = NoisyLinear(2, 8, 0.07)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(8, 8)
        self.a2 = nn.ReLU()
        self.l3 = nn.Linear(8, 1)
        self.a3 = nn.Sigmoid()
    
    def forward(self, x, training=False):
        x = self.l1(x, training)
        x = self.a1(x)
        x = self.l2(x)
        x = self.a2(x)
        x = self.l3(x)
        x = self.a3(x)
        return x
    
    def predict(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        pred = self.forward(x)[:, 0]
        return (pred >= 0.5).float()

# ── Создание модели ──
torch.manual_seed(1)
model = MyNoisyModule()
print('\nАрхитектура модели с NoisyLinear:')
print(model)

# ── Настройка обучения ──
loss_fn = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.015)

train_ds = TensorDataset(x_train, y_train)
batch_size = 2
torch.manual_seed(1)
train_dl = DataLoader(train_ds, batch_size, shuffle=True)

# ── Обучение модели ──
num_epochs = 200
loss_hist_train = [0] * num_epochs
accuracy_hist_train = [0] * num_epochs
loss_hist_valid = [0] * num_epochs
accuracy_hist_valid = [0] * num_epochs

print('\nОбучение модели...')
for epoch in range(num_epochs):
    for x_batch, y_batch in train_dl:
        pred = model(x_batch, True)[:, 0]
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        loss_hist_train[epoch] += loss.item()
        is_correct = ((pred >= 0.5).float() == y_batch).float()
        accuracy_hist_train[epoch] += is_correct.mean()
    
    loss_hist_train[epoch] /= n_train / batch_size
    accuracy_hist_train[epoch] /= n_train / batch_size
    
    pred = model(x_valid)[:, 0]
    loss = loss_fn(pred, y_valid)
    loss_hist_valid[epoch] = loss.item()
    is_correct = ((pred >= 0.5).float() == y_valid).float()
    accuracy_hist_valid[epoch] += is_correct.mean()

print(f'Финальная точность (обучение): {accuracy_hist_train[-1]:.4f}')
print(f'Финальная точность (валидация): {accuracy_hist_valid[-1]:.4f}')

# ── Визуализация результатов ──
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# График потерь
axes[0].plot(loss_hist_train, lw=4, label='Потери при обучении')
axes[0].plot(loss_hist_valid, lw=4, label='Потери при валидации')
axes[0].legend(fontsize=12)
axes[0].set_xlabel('Эпохи', size=12)
axes[0].set_ylabel('Потеря', size=12)
axes[0].set_title('Потери', size=14)

# График точности
axes[1].plot(accuracy_hist_train, lw=4, label='Точность при обучении')
axes[1].plot(accuracy_hist_valid, lw=4, label='Точность при валидации')
axes[1].legend(fontsize=12)
axes[1].set_xlabel('Эпохи', size=12)
axes[1].set_ylabel('Точность', size=12)
axes[1].set_title('Точность', size=14)

# Визуализация разделяющей границы
try:
    from mlxtend.plotting import plot_decision_regions
    plot_decision_regions(X=x_valid.numpy(),
                          y=y_valid.numpy().astype(np.int64),
                          clf=model)
    axes[2].set_xlabel(r'$x_1$', size=12)
    axes[2].set_ylabel(r'$x_2$', size=12)
    axes[2].set_title('Разделяющая граница (NoisyLinear)', size=14)
except ImportError:
    axes[2].text(0.5, 0.5, 'mlxtend не установлен\npip install mlxtend',
                 ha='center', va='center', fontsize=12)
    axes[2].set_title('Разделяющая граница', size=14)

plt.tight_layout()
plt.show()
