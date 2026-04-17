"""
Раздел 4.4. Приведение признаков к одному масштабу
Учебные примеры из книги "Python Machine Learning"
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Простой пример для демонстрации разницы между стандартизацией и нормализацией
print("=== Простой пример: числа от 0 до 5 ===")
ex = np.array([0, 1, 2, 3, 4, 5])
print("Исходные данные:", ex)
print("Стандартизованы:", (ex - ex.mean()) / ex.std())
print("Нормализованы:", (ex - ex.min()) / (ex.max() - ex.min()))
print()

# Таблица 4.1 - Сравнение между стандартизацией и минимаксной нормализацией
print("=== Таблица 4.1. Сравнение методов масштабирования ===")
print("Вход\t\tСтандартизация\tМинимаксная нормализация")
for i, val in enumerate(ex):
    std_val = (val - ex.mean()) / ex.std()
    norm_val = (val - ex.min()) / (ex.max() - ex.min())
    print(f"{val:.1f}\t\t{std_val:.5f}\t\t{norm_val:.1f}")
print()

# Пример использования MinMaxScaler
print("=== Использование MinMaxScaler ===")
# Пример данных с разными масштабами
X_example = np.array([[1, 10000],
                     [2, 20000],
                     [3, 30000],
                     [4, 40000],
                     [5, 50000]])
print("Исходные данные (разные масштабы):")
print(X_example)
print()

mms = MinMaxScaler()
X_norm = mms.fit_transform(X_example)
print("После нормализации (диапазон [0, 1]):")
print(X_norm)
print()

# Пример использования StandardScaler
print("=== Использование StandardScaler ===")
stdsc = StandardScaler()
X_std = stdsc.fit_transform(X_example)
print("После стандартизации (среднее=0, std=1):")
print(X_std)
print()

# Сравнение статистики
print("=== Сравнение статистики ===")
print("Исходные данные:")
print("Среднее:", X_example.mean(axis=0))
print("Стд. отклонение:", X_example.std(axis=0))
print()
print("После стандартизации:")
print("Среднее:", X_std.mean(axis=0))
print("Стд. отклонение:", X_std.std(axis=0))
