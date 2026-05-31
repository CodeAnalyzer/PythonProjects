import numpy as np
import matplotlib.pyplot as plt
import torch

# ── Определение функций активации ──

def logistic(z):
    """Логистическая (сигмоидная) функция"""
    return 1.0 / (1.0 + np.exp(-z))

def tanh(z):
    """Гиперболический тангенс (ручная реализация)"""
    e_p = np.exp(z)
    e_m = np.exp(-z)
    return (e_p - e_m) / (e_p + e_m)

# ── Подготовка данных для графика ──
z = np.arange(-5, 5, 0.005)
log_act = logistic(z)
tanh_act = tanh(z)

# ── Визуализация сравнения ──
plt.figure(figsize=(10, 6))
plt.ylim([-1.5, 1.5])
plt.xlabel('Действ. вход $z$', size=12)
plt.ylabel('Активация $\\phi(z)$', size=12)

# Горизонтальные линии для ориентира
plt.axhline(1, color='black', linestyle=':', alpha=0.5)
plt.axhline(0.5, color='black', linestyle=':', alpha=0.5)
plt.axhline(0, color='black', linestyle='-', alpha=0.3)
plt.axhline(-0.5, color='black', linestyle=':', alpha=0.5)
plt.axhline(-1, color='black', linestyle=':', alpha=0.5)

# Графики функций
plt.plot(z, tanh_act, linewidth=3, linestyle='--', label='tanh')
plt.plot(z, log_act, linewidth=3, label='logistic')

plt.legend(loc='lower right', fontsize=12)
plt.title('Сравнение сигмоидных функций: tanh vs logistic', size=14)
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.show()

# ── Использование встроенных функций NumPy и PyTorch ──

print('\n--- Использование np.tanh() ---')
print(np.tanh(z[:5]))  # первые 5 значений
print('...')
print(np.tanh(z[-5:]))  # последние 5 значений

print('\n--- Использование torch.tanh() ---')
z_torch = torch.from_numpy(z)
tanh_torch = torch.tanh(z_torch)
print(tanh_torch[:5])
print('...')
print(tanh_torch[-5:])

print('\n--- Использование torch.sigmoid() ---')
sigmoid_torch = torch.sigmoid(z_torch)
print(sigmoid_torch[:5])
print('...')
print(sigmoid_torch[-5:])

print('\n--- Диапазоны выходных значений ---')
print(f'logistic: ({log_act.min():.4f}, {log_act.max():.4f})')
print(f'tanh:     ({tanh_act.min():.4f}, {tanh_act.max():.4f})')
