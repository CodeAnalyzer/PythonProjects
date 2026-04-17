import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    """Логистическая сигмоидная функция."""
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    """Производная сигмоидной функции."""
    s = sigmoid(z)
    return s * (1 - s)

print("📈 ВИЗУАЛИЗАЦИЯ СИГМОИДНОЙ ФУНКЦИИ")
print("=" * 50)

# 1. Базовая визуализация сигмоиды
print("\n🎯 1. БАЗОВАЯ СИГМОИДА")
print("-" * 30)

# Создаем значения z от -7 до 7
z = np.arange(-7, 7, 0.1)
sigma_z = sigmoid(z)

print(f"Диапазон z: [{z.min():.1f}, {z.max():.1f}]")
print(f"Диапазон sigmoid(z): [{sigma_z.min():.4f}, {sigma_z.max():.4f}]")
print(f"sigmoid(0) = {sigmoid(0):.4f}")
print(f"sigmoid(-∞) ≈ {sigmoid(-100):.6f}")
print(f"sigmoid(+∞) ≈ {sigmoid(100):.6f}")

# 2. Визуализация
plt.figure(figsize=(15, 10))

# График 1: Основная сигмоида
plt.subplot(2, 3, 1)
plt.plot(z, sigma_z, linewidth=3, color='blue', label='σ(z) = 1/(1+e^(-z))')
plt.axvline(0.0, color='k', linestyle='--', alpha=0.5)
plt.axhline(0.5, color='k', linestyle='--', alpha=0.5)
plt.ylim(-0.1, 1.1)
plt.xlabel('z')
plt.ylabel('σ(z)')
plt.title('Логистическая сигмоидная функция')
plt.yticks([0.0, 0.5, 1.0])
plt.grid(True, alpha=0.3)
plt.legend()

# График 2: Производная сигмоиды
plt.subplot(2, 3, 2)
sigma_z_derivative = sigmoid_derivative(z)
plt.plot(z, sigma_z_derivative, linewidth=3, color='red', label="σ'(z) = σ(z)(1-σ(z))")
plt.axvline(0.0, color='k', linestyle='--', alpha=0.5)
plt.axhline(0.25, color='k', linestyle='--', alpha=0.5)
plt.xlabel('z')
plt.ylabel("σ'(z)")
plt.title('Производная сигмоидной функции')
plt.grid(True, alpha=0.3)
plt.legend()

# График 3: Сравнение с другими функциями активации
plt.subplot(2, 3, 3)
# Сигмоида
plt.plot(z, sigma_z, linewidth=2, label='Сигмоида', color='blue')

# Гиперболический тангенс
tanh_z = np.tanh(z)
plt.plot(z, (tanh_z + 1) / 2, linewidth=2, label='Tanh (нормализованный)', color='green')

# ReLU
relu_z = np.maximum(0, z) / 7  # Нормализуем для сравнения
plt.plot(z, relu_z, linewidth=2, label='ReLU (нормализованный)', color='orange')

plt.axvline(0.0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('z')
plt.ylabel('Активация')
plt.title('Сравнение функций активации')
plt.legend()
plt.grid(True, alpha=0.3)

# График 4: Поведение на крайних значениях
plt.subplot(2, 3, 4)
extreme_z = np.array([-10, -5, -2, -1, 0, 1, 2, 5, 10])
extreme_sigma = sigmoid(extreme_z)

colors = ['red' if x < 0 else 'green' for x in extreme_z]
bars = plt.bar(range(len(extreme_z)), extreme_sigma, color=colors, alpha=0.7)
plt.xticks(range(len(extreme_z)), [f'{z}' for z in extreme_z])
plt.ylabel('σ(z)')
plt.title('Значения сигмоиды на крайних точках')
plt.axhline(0.5, color='k', linestyle='--', alpha=0.5)
plt.grid(True, alpha=0.3, axis='y')

# Добавляем значения на столбцы
for bar, val in zip(bars, extreme_sigma):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{val:.3f}', ha='center', va='bottom', fontsize=8)

# График 5: Влияние масштаба z
plt.subplot(2, 3, 5)
scales = [0.5, 1.0, 2.0, 5.0]
colors_scale = ['blue', 'green', 'orange', 'red']

for scale, color in zip(scales, colors_scale):
    z_scaled = z * scale
    sigma_scaled = sigmoid(z_scaled)
    plt.plot(z, sigma_scaled, linewidth=2, label=f'Масштаб: {scale}x', color=color)

plt.axvline(0.0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('z')
plt.ylabel('σ(scale·z)')
plt.title('Влияние масштабирования входа')
plt.legend()
plt.grid(True, alpha=0.3)

# График 6: Практическое применение - вероятности
plt.subplot(2, 3, 6)
# Симулируем логит-регрессию для бинарной классификации
np.random.seed(42)
n_samples = 100
X_example = np.random.randn(n_samples)
# Истинные веса
w_true, b_true = 2.0, -1.0
z_example = w_true * X_example + b_true
probabilities = sigmoid(z_example)

# Разделяем на классы
y_example = (probabilities > 0.5).astype(int)

# Визуализация
plt.scatter(X_example[y_example == 0], probabilities[y_example == 0], 
           color='red', alpha=0.6, label='Класс 0', s=30)
plt.scatter(X_example[y_example == 1], probabilities[y_example == 1], 
           color='blue', alpha=0.6, label='Класс 1', s=30)

# Сигмоида для визуализации границы
x_range = np.linspace(X_example.min(), X_example.max(), 100)
z_range = w_true * x_range + b_true
prob_range = sigmoid(z_range)
plt.plot(x_range, prob_range, 'k-', linewidth=2, label='σ(w·x + b)')

plt.axhline(0.5, color='k', linestyle='--', alpha=0.5)
plt.xlabel('x')
plt.ylabel('Вероятность класса 1')
plt.title('Логистическая регрессия')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 3. Математический анализ
print("\n🔍 2. МАТЕМАТИЧЕСКИЙ АНАЛИЗ")
print("-" * 30)

print("Свойства сигмоидной функции:")
print("• Область определения: (-∞, +∞)")
print("• Область значений: (0, 1)")
print("• Монотонно возрастающая")
print("• σ(0) = 0.5 (центр симметрии)")
print("• σ(-z) = 1 - σ(z) (симметрия)")

print("\nПределы:")
print("• lim(z→-∞) σ(z) = 0")
print("• lim(z→+∞) σ(z) = 1")
print("• lim(z→0) σ(z) = 0.5")

print("\nПроизводная:")
print("• σ'(z) = σ(z) × (1 - σ(z))")
print("• Максимум производной: σ'(0) = 0.25")
print("• Используется в обратном распространении ошибки")

# 4. Практические примеры
print("\n💡 3. ПРАКТИЧЕСКИЕ ПРИМЕНЕНИЯ")
print("-" * 30)

test_values = [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]
print("z → σ(z) → Интерпретация:")
for z_val in test_values:
    sigma_val = sigmoid(z_val)
    if sigma_val < 0.1:
        interpretation = "Очень низкая вероятность"
    elif sigma_val < 0.3:
        interpretation = "Низкая вероятность"
    elif sigma_val < 0.7:
        interpretation = "Неопределенность"
    elif sigma_val < 0.9:
        interpretation = "Высокая вероятность"
    else:
        interpretation = "Очень высокая вероятность"
    
    print(f"{z_val:4.1f} → {sigma_val:6.3f} → {interpretation}")

print(f"\n🎯 ВЫВОДЫ:")
print("=" * 50)
print("• Сигмоида преобразует любые значения в диапазон [0, 1]")
print("• Идеальна для вероятностной интерпретации")
print("• Гладкая функция - подходит для градиентного спуска")
print("• Основана на экспоненциальной функции")
print("• Используется в логистической регрессии и нейронных сетях")
