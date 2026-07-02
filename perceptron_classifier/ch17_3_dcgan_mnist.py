"""
Раздел 17.3.3: Реализация DCGAN (Deep Convolutional GAN) для MNIST

Генератор: транспонированные свёртки (ConvTranspose2d) + BatchNorm + LeakyReLU
Дискриминатор: свёртки (Conv2d) + BatchNorm + LeakyReLU

DCGAN генерирует изображения более высокого качества, чем обычная GAN
на полносвязных слоях.
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


# ──────────────────────────────────────────────────────────────
# 17.3.3. Генератор и дискриминатор DCGAN
# ──────────────────────────────────────────────────────────────

def make_generator_network(input_size, n_filters):
    """
    Генератор DCGAN.
    input_size: размер вектора z
    n_filters: базовое количество фильтров

    Архитектура (карты признаков):
      z: (input_size, 1, 1)
      -> ConvTranspose2d(input_size, n_filters*4, 4, 1, 0) -> 4x4
      -> ConvTranspose2d(n_filters*4, n_filters*2, 3, 2, 1) -> 7x7
      -> ConvTranspose2d(n_filters*2, n_filters, 4, 2, 1) -> 14x14
      -> ConvTranspose2d(n_filters, 1, 4, 2, 1) -> 28x28
    """
    model = nn.Sequential(
        nn.ConvTranspose2d(input_size, n_filters * 4, 4, 1, 0, bias=False),
        nn.BatchNorm2d(n_filters * 4),
        nn.LeakyReLU(0.2),

        nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 3, 2, 1, bias=False),
        nn.BatchNorm2d(n_filters * 2),
        nn.LeakyReLU(0.2),

        nn.ConvTranspose2d(n_filters * 2, n_filters, 4, 2, 1, bias=False),
        nn.BatchNorm2d(n_filters),
        nn.LeakyReLU(0.2),

        nn.ConvTranspose2d(n_filters, 1, 4, 2, 1, bias=False),
        nn.Tanh()
    )
    return model


class Discriminator(nn.Module):
    """
    Дискриминатор DCGAN.
    n_filters: базовое количество фильтров

    Архитектура:
      1x28x28
      -> Conv2d(1, n_filters, 4, 2, 1) -> 14x14
      -> Conv2d(n_filters, n_filters*2, 4, 2, 1) -> 7x7
      -> Conv2d(n_filters*2, n_filters*4, 3, 2, 1) -> 4x4
      -> Conv2d(n_filters*4, 1, 4, 1, 0) -> 1x1
    """
    def __init__(self, n_filters):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, n_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters * 2, n_filters * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(n_filters * 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        output = self.network(input)
        return output.view(-1, 1).squeeze(0)


# ──────────────────────────────────────────────────────────────
# Настройки и данные
# ──────────────────────────────────────────────────────────────

z_size = 100
image_size = (28, 28)
n_filters = 32
mode_z = 'uniform'
num_epochs = 100
batch_size = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("17.3.3. DCGAN на MNIST")
print("=" * 70)
print(f"Устройство: {device}")

# Загрузка MNIST (та же предобработка, что в 17.2)
image_path = './'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])

mnist_dataset = torchvision.datasets.MNIST(
    root=image_path, train=True,
    transform=transform, download=False
)

mnist_dl = DataLoader(
    mnist_dataset, batch_size=batch_size,
    shuffle=True, drop_last=True
)

# Создание моделей
gen_model = make_generator_network(z_size, n_filters).to(device)
print("\nГенератор:")
print(gen_model)

disc_model = Discriminator(n_filters).to(device)
print("\nДискриминатор:")
print(disc_model)

# Потери и оптимизаторы
loss_fn = nn.BCELoss()
g_optimizer = torch.optim.Adam(gen_model.parameters(), 0.0003)
d_optimizer = torch.optim.Adam(disc_model.parameters(), 0.0002)


# ──────────────────────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────────────────────

def create_noise(batch_size, z_size, mode_z):
    """
    Создание 4D тензора шума для DCGAN.
    Возвращает (batch_size, z_size, 1, 1)
    """
    if mode_z == 'uniform':
        input_z = torch.rand(batch_size, z_size, 1, 1) * 2 - 1
    elif mode_z == 'normal':
        input_z = torch.randn(batch_size, z_size, 1, 1)
    return input_z


def create_samples(g_model, input_z):
    """Генерация примеров из шума. Возвращает изображения в [0, 1]."""
    g_output = g_model(input_z)
    images = torch.reshape(g_output, (batch_size, *image_size))
    return (images + 1) / 2.0


def d_train(x):
    """Обучение дискриминатора на одном батче."""
    disc_model.zero_grad()

    # Реальный батч
    batch_size = x.size(0)
    x = x.to(device)
    d_labels_real = torch.ones(batch_size, 1, device=device)
    d_proba_real = disc_model(x)
    d_loss_real = loss_fn(d_proba_real, d_labels_real)

    # Поддельный батч
    input_z = create_noise(batch_size, z_size, mode_z).to(device)
    g_output = gen_model(input_z)
    d_proba_fake = disc_model(g_output)
    d_labels_fake = torch.zeros(batch_size, 1, device=device)
    d_loss_fake = loss_fn(d_proba_fake, d_labels_fake)

    # Обратное распространение и оптимизация только D
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

    # Обратное распространение и оптимизация только G
    g_loss.backward()
    g_optimizer.step()

    return g_loss.data.item()


# ──────────────────────────────────────────────────────────────
# Обучение DCGAN
# ──────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Обучение DCGAN")
print("=" * 70)

fixed_z = create_noise(batch_size, z_size, mode_z).to(device)
epoch_samples = []

torch.manual_seed(1)
np.random.seed(1)

for epoch in range(1, num_epochs + 1):
    d_losses, g_losses = [], []

    gen_model.train()
    for i, (x, _) in enumerate(mnist_dl):
        d_loss, d_proba_real, d_proba_fake = d_train(x)
        d_losses.append(d_loss)
        g_losses.append(g_train(x))

    print(f'Эпоха {epoch:03d} | Средн. потери>> '
          f'G/D {torch.FloatTensor(g_losses).mean():.4f}/'
          f'{torch.FloatTensor(d_losses).mean():.4f}')

    gen_model.eval()
    epoch_samples.append(
        create_samples(gen_model, fixed_z).detach().cpu().numpy()
    )


# ──────────────────────────────────────────────────────────────
# Визуализация результатов
# ──────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Визуализация результатов")
print("=" * 70)

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
plt.savefig('dcgan_samples.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("Готово! Сгенерированные изображения сохранены в dcgan_samples.png")
print("=" * 70)
