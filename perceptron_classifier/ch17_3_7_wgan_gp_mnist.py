"""
Раздел 17.3.7: Реализация WGAN-GP для обучения модели DCGAN

Wasserstein GAN с градиентным штрафом (Gradient Penalty).
Отличия от обычной DCGAN:
  - InstanceNorm2d вместо BatchNorm2d
  - Функция потерь Вассерштейна вместо BCE
  - Градиентный штраф для стабилизации обучения
  - Критик обучается 5 раз за одну итерацию генератора
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.autograd import grad as torch_grad


# ──────────────────────────────────────────────────────────────
# Модели WGAN (InstanceNorm вместо BatchNorm)
# ──────────────────────────────────────────────────────────────

def make_generator_network_wgan(input_size, n_filters):
    """
    Генератор WGAN с InstanceNorm2d вместо BatchNorm2d.
    """
    model = nn.Sequential(
        nn.ConvTranspose2d(input_size, n_filters * 4, 4, 1, 0, bias=False),
        nn.InstanceNorm2d(n_filters * 4),
        nn.LeakyReLU(0.2),

        nn.ConvTranspose2d(n_filters * 4, n_filters * 2, 3, 2, 1, bias=False),
        nn.InstanceNorm2d(n_filters * 2),
        nn.LeakyReLU(0.2),

        nn.ConvTranspose2d(n_filters * 2, n_filters, 4, 2, 1, bias=False),
        nn.InstanceNorm2d(n_filters),
        nn.LeakyReLU(0.2),

        nn.ConvTranspose2d(n_filters, 1, 4, 2, 1, bias=False),
        nn.Tanh()
    )
    return model


class DiscriminatorWGAN(nn.Module):
    """
    Дискриминатор (критик) WGAN с InstanceNorm2d.
    """
    def __init__(self, n_filters):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, n_filters, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters, n_filters * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(n_filters * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(n_filters * 2, n_filters * 4, 3, 2, 1, bias=False),
            nn.InstanceNorm2d(n_filters * 4),
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
lambda_gp = 10.0
critic_iterations = 5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print("=" * 70)
print("17.3.7. WGAN-GP на MNIST")
print("=" * 70)
print(f"Устройство: {device}")

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

mnist_dl = DataLoader(
    mnist_dataset, batch_size=batch_size,
    shuffle=True, drop_last=True
)

# Создание моделей
gen_model = make_generator_network_wgan(z_size, n_filters).to(device)
print("\nГенератор (WGAN):")
print(gen_model)

disc_model = DiscriminatorWGAN(n_filters).to(device)
print("\nДискриминатор (WGAN):")
print(disc_model)

# Оптимизаторы
g_optimizer = torch.optim.Adam(gen_model.parameters(), 0.0002)
d_optimizer = torch.optim.Adam(disc_model.parameters(), 0.0002)


# ──────────────────────────────────────────────────────────────
# Вспомогательные функции
# ──────────────────────────────────────────────────────────────

def create_noise(batch_size, z_size, mode_z):
    """Создание 4D тензора шума (batch_size, z_size, 1, 1)."""
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


def gradient_penalty(real_data, generated_data):
    """
    Градиентный штраф для WGAN-GP.
    Штраф заставляет градиент дискриминатора иметь норму ~1
    на интерполированных точках между реальными и сгенерированными данными.
    """
    batch_size = real_data.size(0)

    # Вычисление интерполяции
    alpha = torch.rand(real_data.shape[0], 1, 1, 1,
                       requires_grad=True, device=device)
    interpolated = alpha * real_data + (1 - alpha) * generated_data

    # Вычисление вероятности интерполированных примеров
    proba_interpolated = disc_model(interpolated)

    # Вычисление градиентов вероятностей
    gradients = torch_grad(
        outputs=proba_interpolated, inputs=interpolated,
        grad_outputs=torch.ones(proba_interpolated.size(), device=device),
        create_graph=True, retain_graph=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradients_norm = gradients.norm(2, dim=1)

    return lambda_gp * ((gradients_norm - 1) ** 2).mean()


def d_train_wgan(x):
    """Обучение дискриминатора (критика) WGAN."""
    disc_model.zero_grad()
    batch_size = x.size(0)
    x = x.to(device)

    # Вычисление вероятностей реальных и сгенерированных данных
    d_real = disc_model(x)
    input_z = create_noise(batch_size, z_size, mode_z).to(device)
    g_output = gen_model(input_z)
    d_generated = disc_model(g_output)

    # Потери Вассерштейна + градиентный штраф
    d_loss = d_generated.mean() - d_real.mean() + \
        gradient_penalty(x.data, g_output.data)

    d_loss.backward()
    d_optimizer.step()

    return d_loss.data.item()


def g_train_wgan(x):
    """Обучение генератора WGAN."""
    gen_model.zero_grad()
    batch_size = x.size(0)
    input_z = create_noise(batch_size, z_size, mode_z).to(device)

    g_output = gen_model(input_z)
    d_generated = disc_model(g_output)

    # Генератор максимизирует d_generated (минимизирует -d_generated)
    g_loss = -d_generated.mean()

    # Обратное распространение и оптимизация только G
    g_loss.backward()
    g_optimizer.step()

    return g_loss.data.item()


# ──────────────────────────────────────────────────────────────
# Обучение WGAN-GP
# ──────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("Обучение WGAN-GP")
print(f"critic_iterations: {critic_iterations}, lambda_gp: {lambda_gp}")
print("=" * 70)

fixed_z = create_noise(batch_size, z_size, mode_z).to(device)
epoch_samples_wgan = []

torch.manual_seed(1)
np.random.seed(1)

for epoch in range(1, num_epochs + 1):
    gen_model.train()
    d_losses, g_losses = [], []

    for i, (x, _) in enumerate(mnist_dl):
        # Критик обучается critic_iterations раз за одну итерацию генератора
        for _ in range(critic_iterations):
            d_loss = d_train_wgan(x)
            d_losses.append(d_loss)

        g_losses.append(g_train_wgan(x))

    print(f'Эпоха {epoch:03d} | Потери D >> '
          f'{torch.FloatTensor(d_losses).mean():.4f}')

    gen_model.eval()
    epoch_samples_wgan.append(
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
        image = epoch_samples_wgan[e - 1][j]
        ax.imshow(image, cmap='gray_r')

plt.tight_layout()
plt.savefig('wgan_gp_samples.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 70)
print("Готово! Изображения сохранены в wgan_gp_samples.png")
print("=" * 70)
