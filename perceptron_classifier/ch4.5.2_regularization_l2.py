"""
Раздел 4.5.2. Геометрическая интерпретация регуляризации L2
Учебные примеры из книги "Python Machine Learning"
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches

# Настройка стиля графиков
plt.style.use('seaborn-v0_8-darkgrid')

# Создаем сетку для весовых коэффициентов w1 и w2
w1 = np.linspace(-3, 3, 100)
w2 = np.linspace(-3, 3, 100)
W1, W2 = np.meshgrid(w1, w2)

# Определяем функцию потерь MSE (сферическая форма для простоты)
# Предполагаем, что минимум находится в точке (2, 2)
loss = (W1 - 2)**2 + (W2 - 2)**2

# Создаем фигуру с двумя подграфиками
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Левый график: функция потерь без регуляризации (Рис. 4.5)
print("=== Рис. 4.5: Контуры функции потерь MSE ===")
contour1 = ax1.contour(W1, W2, loss, levels=20, cmap='viridis')
ax1.clabel(contour1, inline=True, fontsize=10)
ax1.scatter([2], [2], color='red', s=100, zorder=5, label='Минимум функции потерь')
ax1.set_xlabel('w1')
ax1.set_ylabel('w2')
ax1.set_title('Функция потерь MSE без регуляризации')
ax1.legend()
ax1.grid(True)
ax1.axis('equal')

# Правый график: функция потерь с регуляризацией L2 (Рис. 4.6)
print("=== Рис. 4.6: Регуляризация L2 ===")
contour2 = ax2.contour(W1, W2, loss, levels=20, cmap='viridis', alpha=0.7)
ax2.clabel(contour2, inline=True, fontsize=10)

# Область регуляризации L2 (шар)
# L2 ограничение: w1^2 + w2^2 <= C
l2_radius = 1.5  # радиус шара L2
l2_circle = patches.Circle((0, 0), l2_radius, 
                          edgecolor='red', facecolor='red', 
                          alpha=0.3, label='Область L2')
ax2.add_patch(l2_circle)

# Оптимальная точка (пересечение контура и шара L2)
# Это точка на контуре, ближайшая к началу координат
optimal_w1 = 2 * l2_radius / np.sqrt(2**2 + 2**2)
optimal_w2 = 2 * l2_radius / np.sqrt(2**2 + 2**2)
ax2.scatter([optimal_w1], [optimal_w2], color='black', s=100, zorder=5, 
           label='Оптимальная точка с L2')
ax2.scatter([2], [2], color='red', s=100, zorder=5, marker='x',
           label='Минимум без регуляризации')

ax2.set_xlabel('w1')
ax2.set_ylabel('w2')
ax2.set_title('Функция потерь MSE с регуляризацией L2')
ax2.legend()
ax2.grid(True)
ax2.axis('equal')

plt.tight_layout()
plt.savefig('regularization_l2_visualization.png', dpi=150, bbox_inches='tight')
print("График сохранен как 'regularization_l2_visualization.png'")
plt.show()

# Демонстрация влияния параметра регуляризации
print("\n=== Влияние параметра регуляризации λ ===")
print("При увеличении λ:")
print("- Шар L2 становится меньше (радиус уменьшается)")
print("- Веса сдвигаются ближе к нулю")
print("- Модель становится более простой (меньше переобучение)")
print()

# Пример с разными значениями регуляризации
fig2, ax = plt.subplots(figsize=(8, 8))
contour3 = ax.contour(W1, W2, loss, levels=20, cmap='viridis', alpha=0.5)
ax.clabel(contour3, inline=True, fontsize=8)

# Разные радиусы шара L2 для разных значений λ
radii = [2.5, 1.8, 1.2, 0.6]
colors = ['green', 'blue', 'orange', 'red']
labels = ['λ=0.1', 'λ=0.5', 'λ=1.0', 'λ=2.0']

for radius, color, label in zip(radii, colors, labels):
    circle = patches.Circle((0, 0), radius, 
                          edgecolor=color, facecolor=color, 
                          alpha=0.15, label=label)
    ax.add_patch(circle)

ax.scatter([2], [2], color='black', s=100, zorder=5, marker='x',
           label='Минимум без регуляризации')
ax.set_xlabel('w1')
ax.set_ylabel('w2')
ax.set_title('Влияние параметра регуляризации λ на размер области L2')
ax.legend()
ax.grid(True)
ax.axis('equal')

plt.tight_layout()
plt.savefig('regularization_lambda_effect.png', dpi=150, bbox_inches='tight')
print("График влияния λ сохранен как 'regularization_lambda_effect.png'")
plt.show()

# Численный пример: влияние L2 на веса
print("\n=== Численный пример: влияние L2 регуляризации ===")
print("Предположим, оптимальные веса без регуляризации: w1=2, w2=2")
print("С регуляризацией L2 веса уменьшаются пропорционально")
print()

lambda_values = [0, 0.1, 0.5, 1.0, 2.0, 5.0]
w1_orig, w2_orig = 2.0, 2.0

print("λ\t\tw1\t\tw2\t\t||w||")
for lam in lambda_values:
    # Упрощенная формула: w_reg = w_orig / (1 + lambda)
    # Это демонстрационный пример, реальная формула зависит от алгоритма
    w1_reg = w1_orig / (1 + lam)
    w2_reg = w2_orig / (1 + lam)
    norm = np.sqrt(w1_reg**2 + w2_reg**2)
    print(f"{lam:.1f}\t\t{w1_reg:.3f}\t\t{w2_reg:.3f}\t\t{norm:.3f}")
