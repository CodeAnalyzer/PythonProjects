"""
Раздел 17.2: Реализация обычной GAN на полносвязных слоях для генерации
изображений MNIST.

Подразделы:
  17.2.2. Реализация сетей генератора и дискриминатора
  17.2.3. Определение набора обучающих данных
  17.2.4. Обучение модели GAN
"""

import itertools
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


# ──────────────────────────────────────────────────────────────
# 17.2.2. Реализация сетей генератора и дискриминатора
# ──────────────────────────────────────────────────────────────

def make_generator_network(
        input_size=20,
        num_hidden_layers=1,
        num_hidden_units=100,
        num_output_units=784):
    """
    Создание сети генератора.
    Скрытые слои: Linear + LeakyReLU
    Выходной слой: Linear + Tanh (значения в [-1, 1])
    """
    model = nn.Sequential()
    for i in range(num_hidden_layers):
        model.add_module(
            f'fc_g{i}',
            nn.Linear(input_size, num_hidden_units, bias=False)
        )
        model.add_module(f'relu_g{i}', nn.LeakyReLU())
        input_size = num_hidden_units

    model.add_module(
        f'fc_g{num_hidden_layers}',
        nn.Linear(input_size, num_output_units)
    )
    model.add_module('tanh_g', nn.Tanh())
    return model


def make_discriminator_network(
        input_size,
        num_hidden_layers=1,
        num_hidden_units=100,
        num_output_units=1):
    """
    Создание сети дискриминатора.
    Скрытые слои: Linear(без bias) + LeakyReLU + Dropout
    Выходной слой: Linear + Sigmoid (вероятности)
    """
    model = nn.Sequential()
    for i in range(num_hidden_layers):
        model.add_module(
            f'fc_d{i}',
            nn.Linear(input_size, num_hidden_units, bias=False)
        )
        model.add_module(f'relu_d{i}', nn.LeakyReLU())
        model.add_module('dropout', nn.Dropout(p=0.5))
        input_size = num_hidden_units

    model.add_module(
        f'fc_d{num_hidden_layers}',
        nn.Linear(input_size, num_output_units)
    )
    model.add_module('sigmoid', nn.Sigmoid())
    return model


# ──────────────────────────────────────────────────────────────
# 17.2.3. Определение набора обучающих данных
# ──────────────────────────────────────────────────────────────

image_size = (28, 28)
z_size = 20
gen_hidden_layers = 1
gen_hidden_size = 100
disc_hidden_layers = 1
disc_hidden_size = 100

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("17.2.2–17.2.4. Обычная GAN на MNIST")
print("=" * 70)
print(f"Устройство: {device}")

# Инициализация моделей
torch.manual_seed(1)
gen_model = make_generator_network(
    input_size=z_size,
    num_hidden_layers=gen_hidden_layers,
    num_hidden_units=gen_hidden_size,
    num_output_units=np.prod(image_size)
)
print("\nГенератор:")
print(gen_model)

disc_model = make_discriminator_network(
    input_size=np.prod(image_size),
    num_hidden_layers=disc_hidden_layers,
    num_hidden_units=disc_hidden_size
)
print("\nДискриминатор:")
print(disc_model)

# Загрузка MNIST
image_path = './'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])

mnist_dataset = torchvision.datasets.MNIST(
    root=image_path, train=True,
    transform=transform, download=False
)

example, label = next(iter(mnist_dataset))
print(f"\nMNIST пример -- Min: {example.min():.1f} Max: {example.max():.1f}")
print(f"Размер: {example.shape}")


def create_noise(batch_size, z_size, mode_z):
    """
    Создание случайного вектора z.
    mode_z: 'uniform' или 'normal'
    """
    if mode_z == 'uniform':
        input_z = torch.rand(batch_size, z_size) * 2 - 1
    elif mode_z == 'normal':
        input_z = torch.randn(batch_size, z_size)
    return input_z


# Проверка потока данных
batch_size = 32
dataloader = DataLoader(mnist_dataset, batch_size, shuffle=False)
input_real, label = next(iter(dataloader))
input_real = input_real.view(batch_size, -1)

torch.manual_seed(1)
mode_z = 'uniform'
input_z = create_noise(batch_size, z_size, mode_z)

print(f'\nвход-z -- размеры: {input_z.shape}')
print(f'вход-реальн. -- размеры: {input_real.shape}')

g_output = gen_model(input_z)
print(f'Выход G -- размеры: {g_output.shape}')

d_proba_real = disc_model(input_real)
d_proba_fake = disc_model(g_output)
print(f'Дискр. (реальный) -- размеры: {d_proba_real.shape}')
print(f'Дискр. (фиктивный) -- размеры: {d_proba_fake.shape}')

# Демонстрация вычисления потерь
loss_fn = nn.BCELoss()

# Потери генератора
g_labels_real = torch.ones_like(d_proba_fake)
g_loss = loss_fn(d_proba_fake, g_labels_real)
print(f'\nПотери генератора: {g_loss:.4f}')

# Потери дискриминатора
d_labels_real = torch.ones_like(d_proba_real)
d_labels_fake = torch.zeros_like(d_proba_fake)
d_loss_real = loss_fn(d_proba_real, d_labels_real)
d_loss_fake = loss_fn(d_proba_fake, d_labels_fake)
print(f'Потери дискриминатора: Реальн. {d_loss_real:.4f} '
      f'Поддельн. {d_loss_fake:.4f}')


# ──────────────────────────────────────────────────────────────
# 17.2.4. Обучение модели GAN
# ──────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Обучение GAN")
print("=" * 70)

batch_size = 64
torch.manual_seed(1)
np.random.seed(1)

mnist_dl = DataLoader(
    mnist_dataset, batch_size=batch_size,
    shuffle=True, drop_last=True
)

gen_model = make_generator_network(
    input_size=z_size,
    num_hidden_layers=gen_hidden_layers,
    num_hidden_units=gen_hidden_size,
    num_output_units=np.prod(image_size)
).to(device)

disc_model = make_discriminator_network(
    input_size=np.prod(image_size),
    num_hidden_layers=disc_hidden_layers,
    num_hidden_units=disc_hidden_size
).to(device)

loss_fn = nn.BCELoss()
g_optimizer = torch.optim.Adam(gen_model.parameters())
d_optimizer = torch.optim.Adam(disc_model.parameters())


def d_train(x):
    """Обучение дискриминатора на одном батче."""
    disc_model.zero_grad()

    # Обучение на реальном батче
    batch_size = x.size(0)
    x = x.view(batch_size, -1).to(device)
    d_labels_real = torch.ones(batch_size, 1, device=device)
    d_proba_real = disc_model(x)
    d_loss_real = loss_fn(d_proba_real, d_labels_real)

    # Обучение на поддельном батче
    input_z = create_noise(batch_size, z_size, mode_z).to(device)
    g_output = gen_model(input_z)
    d_proba_fake = disc_model(g_output)
    d_labels_fake = torch.zeros(batch_size, 1, device=device)
    d_loss_fake = loss_fn(d_proba_fake, d_labels_fake)

    # Обратное распространение и оптимизация ТОЛЬКО параметров D
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    d_optimizer.step()

    return d_loss.data.item(), d_proba_real.detach(), d_proba_fake.detach()


def g_train(x):
    """Обучение генератора на одном батче."""
    gen_model.zero_grad()

    batch_size = x.size(0)
    input_z = create_noise(batch_size, z_size, mode_z).to(device)
    g_labels_real = torch.ones(batch_size, 1, device=device)

    g_output = gen_model(input_z)
    d_proba_fake = disc_model(g_output)
    g_loss = loss_fn(d_proba_fake, g_labels_real)

    # Обратное распространение и оптимизация ТОЛЬКО параметров G
    g_loss.backward()
    g_optimizer.step()

    return g_loss.data.item()


# Фиксированный шум для визуализации прогресса
fixed_z = create_noise(batch_size, z_size, mode_z).to(device)


def create_samples(g_model, input_z):
    """Генерация примеров изображений из шума.
    Возвращает изображения в диапазоне [0, 1] для отображения.
    """
    g_output = g_model(input_z)
    images = torch.reshape(g_output, (batch_size, *image_size))
    return (images + 1) / 2.0


# Сборщики метрик
epoch_samples = []
all_d_losses = []
all_g_losses = []
all_d_real = []
all_d_fake = []

num_epochs = 100

for epoch in range(1, num_epochs + 1):
    d_losses, g_losses = [], []
    d_vals_real, d_vals_fake = [], []

    for i, (x, _) in enumerate(mnist_dl):
        d_loss, d_proba_real, d_proba_fake = d_train(x)
        d_losses.append(d_loss)
        g_losses.append(g_train(x))
        d_vals_real.append(d_proba_real.mean().cpu())
        d_vals_fake.append(d_proba_fake.mean().cpu())

    all_d_losses.append(torch.tensor(d_losses).mean())
    all_g_losses.append(torch.tensor(g_losses).mean())
    all_d_real.append(torch.tensor(d_vals_real).mean())
    all_d_fake.append(torch.tensor(d_vals_fake).mean())

    print(f'Эпоха {epoch:03d} | Средн. потери>> '
          f'G/D {all_g_losses[-1]:.4f}/{all_d_losses[-1]:.4f} '
          f'[D-Real: {all_d_real[-1]:.4f}'
          f' D-Fake: {all_d_fake[-1]:.4f}]')

    epoch_samples.append(
        create_samples(gen_model, fixed_z).detach().cpu().numpy()
    )


# ──────────────────────────────────────────────────────────────
# Визуализация результатов
# ──────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Визуализация результатов")
print("=" * 70)

fig = plt.figure(figsize=(16, 6))

# График потерь
ax = fig.add_subplot(1, 2, 1)
plt.plot(all_g_losses, label='Потери генератора')
half_d_losses = [all_d_loss / 2 for all_d_loss in all_d_losses]
plt.plot(half_d_losses, label='Потери дискриминатора')
plt.legend(fontsize=20)
ax.set_xlabel('Итерация', size=15)
ax.set_ylabel('Потери', size=15)

# График выхода дискриминатора
ax = fig.add_subplot(1, 2, 2)
plt.plot(all_d_real, label=r'Real: $D(\mathbf{x})$')
plt.plot(all_d_fake, label=r'Fake: $D(G(\mathbf{z}))$')
plt.legend(fontsize=20)
ax.set_xlabel('Итерация', size=15)
ax.set_ylabel('Выход дискриминатора', size=15)

plt.tight_layout()
plt.savefig('gan_losses.png', dpi=150, bbox_inches='tight')
plt.show()

# Визуализация сгенерированных изображений на выбранных эпохах
selected_epochs = [1, 2, 4, 10, 50, 100]
fig = plt.figure(figsize=(10, 14))
for i, e in enumerate(selected_epochs):
    for j in range(5):
        ax = fig.add_subplot(6, 5, i * 5 + j + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.text(
                -0.06, 0.5, f'Epoch {e}',
                rotation=90, size=18, color='red',
                horizontalalignment='right',
                verticalalignment='center',
                transform=ax.transAxes
            )
        image = epoch_samples[e - 1][j]
        ax.imshow(image, cmap='gray_r')

plt.tight_layout()
plt.savefig('gan_samples.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("Готово! Графики сохранены в gan_losses.png и gan_samples.png")
print("=" * 70)
