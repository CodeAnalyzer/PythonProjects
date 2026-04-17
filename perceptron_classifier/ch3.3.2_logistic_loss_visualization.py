import matplotlib.pyplot as plt
import numpy as np

def sigmoid(z):
    """Логистическая сигмоидная функция."""
    return 1.0 / (1.0 + np.exp(-z))

def loss_1(z):
    """Логистическая потеря для класса y=1."""
    return -np.log(sigmoid(z))

def loss_0(z):
    """Логистическая потеря для класса y=0."""
    return -np.log(1 - sigmoid(z))

def logistic_loss(sigma, y_true):
    """Логистическая потеря как функция от σ(z) и истинной метки."""
    return - (y_true * np.log(sigma) + (1 - y_true) * np.log(1 - sigma))

print("📉 ВИЗУАЛИЗАЦИЯ ЛОГИСТИЧЕСКИХ ПОТЕРЬ")
print("=" * 50)

# 1. Базовая визуализация потерь
print("\n🎯 1. ЛОГИСТИЧЕСКИЕ ПОТЕРИ")
print("-" * 30)

# Создаем значения z от -10 до 10
z = np.arange(-10, 10, 0.1)
sigma_z = sigmoid(z)

# Вычисляем потери для обоих классов
loss_1_values = [loss_1(x) for x in z]
loss_0_values = [loss_0(x) for x in z]

print(f"Диапазон z: [{z.min():.1f}, {z.max():.1f}]")
print(f"Диапазон σ(z): [{sigma_z.min():.6f}, {sigma_z.max():.6f}]")
print(f"Диапазон потерь (y=1): [{min(loss_1_values):.3f}, {max(loss_1_values):.3f}]")
print(f"Диапазон потерь (y=0): [{min(loss_0_values):.3f}, {max(loss_0_values):.3f}]")

# Проверка крайних случаев
print(f"\nКрайние случаи:")
print(f"σ(z) ≈ 0, y=1: потеря = {loss_1(-10):.3f} (очень большая)")
print(f"σ(z) ≈ 1, y=1: потеря = {loss_1(10):.6f} (почти ноль)")
print(f"σ(z) ≈ 0, y=0: потеря = {loss_0(-10):.6f} (почти ноль)")
print(f"σ(z) ≈ 1, y=0: потеря = {loss_0(10):.3f} (очень большая)")

# 2. Визуализация
plt.figure(figsize=(15, 10))

# График 1: Основные логистические потери
plt.subplot(2, 3, 1)
plt.plot(sigma_z, loss_1_values, linewidth=3, label='L(w,b) если y=1', color='blue')
plt.plot(sigma_z, loss_0_values, linewidth=3, linestyle='--', label='L(w,b) если y=0', color='red')
plt.ylim(0.0, 5.1)
plt.xlim([0, 1])
plt.xlabel('σ(z)')
plt.ylabel('L(w,b)')
plt.title('Логистические потери')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

# График 2: Объединенная функция потерь
plt.subplot(2, 3, 2)
# Создаем примеры с разными истинными метками
sigma_range = np.linspace(0.001, 0.999, 1000)

# Потери для y=1
loss_y1 = logistic_loss(sigma_range, 1)
plt.plot(sigma_range, loss_y1, linewidth=3, label='y=1', color='blue')

# Потери для y=0  
loss_y0 = logistic_loss(sigma_range, 0)
plt.plot(sigma_range, loss_y0, linewidth=3, label='y=0', color='red')

plt.xlabel('σ(z)')
plt.ylabel('L(w,b)')
plt.title('Логистическая функция потерь')
plt.legend()
plt.grid(True, alpha=0.3)

# График 3: Сравнение с другими функциями потерь
plt.subplot(2, 3, 3)
# Логистические потери
plt.plot(sigma_range, loss_y1, linewidth=2, label='Логистическая (y=1)', color='blue')

# MSE (среднеквадратичная ошибка) для сравнения
mse_y1 = (sigma_range - 1)**2
plt.plot(sigma_range, mse_y1, linewidth=2, label='MSE (y=1)', color='green', linestyle='--')

# Hinge loss (SVM) для сравнения
hinge_y1 = np.maximum(0, 1 - sigma_range)
plt.plot(sigma_range, hinge_y1, linewidth=2, label='Hinge (y=1)', color='orange', linestyle=':')

plt.xlabel('σ(z)')
plt.ylabel('Потери')
plt.title('Сравнение функций потерь')
plt.legend()
plt.grid(True, alpha=0.3)

# График 4: Градиенты логистических потерь
plt.subplot(2, 3, 4)
# Вычисляем градиенты
grad_1 = sigmoid(z) * (1 - sigmoid(z))  # dL/dz для y=1
grad_0 = -sigmoid(z) * (1 - sigmoid(z))  # dL/dz для y=0

plt.plot(sigma_z, grad_1, linewidth=3, label='∇L (y=1)', color='blue')
plt.plot(sigma_z, grad_0, linewidth=3, label='∇L (y=0)', color='red')
plt.axhline(0, color='k', linestyle='--', alpha=0.5)
plt.xlabel('σ(z)')
plt.ylabel('Градиент ∂L/∂z')
plt.title('Градиенты логистических потерь')
plt.legend()
plt.grid(True, alpha=0.3)

# График 5: Поведение при разных вероятностях
plt.subplot(2, 3, 5)
# Создаем сценарии с разными истинными метками
np.random.seed(42)
n_scenarios = 10

for i in range(n_scenarios):
    # Случайное истинное значение (0 или 1)
    y_true = np.random.choice([0, 1])
    
    # Случайное предсказанное значение
    sigma_pred = np.random.uniform(0.1, 0.9)
    
    # Вычисляем потерю
    loss_val = logistic_loss(sigma_pred, y_true)
    
    color = 'blue' if y_true == 1 else 'red'
    marker = 'o' if y_true == 1 else 's'
    
    plt.scatter(sigma_pred, loss_val, color=color, marker=marker, s=100, alpha=0.7)
    plt.text(sigma_pred + 0.02, loss_val, f'y={y_true}', fontsize=8)

plt.xlabel('Предсказанная вероятность σ(z)')
plt.ylabel('Потери L(w,b)')
plt.title('Потери для конкретных предсказаний')
plt.grid(True, alpha=0.3)

# График 6: Интуитивное объяснение
plt.subplot(2, 3, 6)
# Создаем интуитивную визуализацию
scenarios = [
    (0.95, 1, "Правильно! ✅"),
    (0.05, 0, "Правильно! ✅"), 
    (0.05, 1, "Ошибка! ❌"),
    (0.95, 0, "Ошибка! ❌"),
    (0.5, 1, "Неуверенно 🤔"),
    (0.5, 0, "Неуверенно 🤔")
]

for sigma_pred, y_true, label in scenarios:
    loss_val = logistic_loss(sigma_pred, y_true)
    color = 'green' if loss_val < 1 else 'orange' if loss_val < 2 else 'red'
    
    plt.bar(label, loss_val, color=color, alpha=0.7)
    plt.text(label, loss_val + 0.1, f'{loss_val:.2f}', ha='center', fontsize=8)

plt.ylabel('Потери')
plt.title('Интуитивное понимание потерь')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# 3. Математический анализ
print("\n🔍 2. МАТЕМАТИЧЕСКИЙ АНАЛИЗ")
print("-" * 30)

print("Логистическая функция потерь:")
print("L(w,b) = -[y·log(σ(z)) + (1-y)·log(1-σ(z))]")
print("где z = w·x + b, σ(z) = 1/(1+e^(-z))")

print("\nДля y = 1:")
print("L(w,b) = -log(σ(z))")
print("• σ(z) → 1: L → 0 (правильное предсказание)")
print("• σ(z) → 0: L → ∞ (неправильное предсказание)")

print("\nДля y = 0:")
print("L(w,b) = -log(1-σ(z))")
print("• σ(z) → 0: L → 0 (правильное предсказание)")
print("• σ(z) → 1: L → ∞ (неправильное предсказание)")

print("\nГрадиенты:")
print("∂L/∂z = σ(z) - y")
print("• Если y=1 и σ(z)<1: положительный градиент")
print("• Если y=0 и σ(z)>0: отрицательный градиент")

# 4. Практическая интерпретация
print("\n💡 3. ПРАКТИЧЕСКАЯ ИНТЕРПРЕТАЦИЯ")
print("-" * 30)

print("Интуитивное понимание:")
print("• Низкие потери ← уверенные правильные предсказания")
print("• Высокие потери ← уверенные неправильные предсказания")
print("• Средние потери ← неуверенные предсказания")

print("\nПреимущества перед MSE:")
print("• Большие штрафы за уверенные ошибки")
print("• Выпуклая функция → глобальный минимум")
print("• Вероятностная интерпретация")

print(f"\n🎯 ВЫВОДЫ:")
print("=" * 50)
print("• Логистические потери измеряют 'расстояние' до истинной метки")
print("• Правильные предсказания → низкие потери")
print("• Неправильные предсказания → высокие потери")
print("• Градиент всегда указывает направление к правильному предсказанию")
print("• Основа для обучения логистической регрессии")
