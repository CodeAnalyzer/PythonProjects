import numpy as np
import pandas as pd
from ch2_3_2_adaline import AdalineGD
import matplotlib.pyplot as plt

# Загрузка и подготовка данных
df = pd.read_csv('iris.data', header=None, encoding='utf-8')
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
X = df.iloc[0:100, [0, 2]].values

# Стандартизация
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

# Обучение Adaline
ada_gd = AdalineGD(n_iter=1000, eta=0.01)
ada_gd.fit(X_std, y)

print("Анализ разделяющей границы:")
print("=" * 50)
print(f"Веса: {ada_gd.w_}")
print(f"Смещение: {ada_gd.b_}")

# Уравнение разделяющей линии: w1*x1 + w2*x2 + b = 0
# x2 = (-w1*x1 - b) / w2
w1, w2 = ada_gd.w_
b = ada_gd.b_

print(f"\nУравнение разделяющей линии:")
print(f"{w1:.3f} * x1 + {w2:.3f} * x2 + {b:.3f} = 0")
print(f"x2 = ({-w1:.3f} * x1 - {b:.3f}) / {w2:.3f}")

# Проверяем несколько точек на границе
print(f"\nПроверка точек на разделяющей границе:")
test_points = [
    [-2.0, 0.0],   # левая сторона
    [0.0, 0.0],    # центр
    [2.0, 0.0],    # правая сторона
    [0.0, -2.0],   # нижняя сторона
    [0.0, 2.0],    # верхняя сторона
]

for point in test_points:
    net_input = w1 * point[0] + w2 * point[1] + b
    prediction = 1 if net_input >= 0.0 else 0
    print(f"Точка {point}: net_input={net_input:.3f}, предсказание={prediction}")

# Анализ ошибочных образцов
predictions = ada_gd.predict(X_std)
misclassified = np.where(predictions != y)[0]

print(f"\nАнализ ошибочных образцов:")
print(f"Всего ошибок: {len(misclassified)}")

if len(misclassified) > 0:
    print(f"\nПервые 5 ошибочных образцов:")
    for i in misclassified[:5]:
        x1, x2 = X_std[i]
        net_input = w1 * x1 + w2 * x2 + b
        true_class = y[i]
        pred_class = predictions[i]
        print(f"Образец {i}: [{x1:.3f}, {x2:.3f}]")
        print(f"  net_input = {w1:.3f}*{x1:.3f} + {w2:.3f}*{x2:.3f} + {b:.3f} = {net_input:.3f}")
        print(f"  Истинный: {true_class}, Предсказанный: {pred_class}")
        print(f"  Проверка: {net_input:.3f} >= 0.0 = {net_input >= 0.0}")

# Визуализация с разделяющей линией
plt.figure(figsize=(10, 8))

# Отображаем данные
plt.scatter(X_std[y == 0, 0], X_std[y == 0, 1], 
           color='red', marker='o', label='Class 0 (Setosa)', alpha=0.8)
plt.scatter(X_std[y == 1, 0], X_std[y == 1, 1], 
           color='blue', marker='s', label='Class 1 (Versicolor)', alpha=0.8)

# Рисуем разделяющую линию
x1_line = np.array([X_std[:, 0].min() - 1, X_std[:, 0].max() + 1])
x2_line = (-w1 * x1_line - b) / w2
plt.plot(x1_line, x2_line, 'g-', linewidth=2, label='Разделяющая линия')

# Отмечаем ошибочные образцы
if len(misclassified) > 0:
    plt.scatter(X_std[misclassified, 0], X_std[misclassified, 1], 
               facecolors='none', edgecolors='yellow', s=100, linewidth=2,
               label='Ошибочные образцы')

plt.xlabel('Длина чашелистика [стандартизирована]')
plt.ylabel('Длина лепестка [стандартизирована]')
plt.title('Анализ разделяющей границы Adaline')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.show()

# Статистика по классам
print(f"\nСтатистика по классам:")
print(f"Class 0 (Setosa): {np.sum(y == 0)} образцов")
print(f"Class 1 (Versicolor): {np.sum(y == 1)} образцов")

print(f"\nТочность по классам:")
for cls in [0, 1]:
    mask = y == cls
    acc = np.mean(predictions[mask] == y[mask]) * 100
    print(f"Class {cls}: {acc:.1f}% ({np.sum(predictions[mask] == y[mask])}/{np.sum(mask)})")
