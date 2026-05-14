"""
Глава 9.4: Обучение устойчивой регрессионной модели с использованием RANSAC

RANSAC (RANdom SAmple Consensus) - алгоритм устойчивой регрессии, который обучает
модель на инлаерах (inliers) - не искаженных выбросами примерах.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RANSACRegressor

# Загружаем очищенные данные
df = pd.read_csv('ames_housing_clean.csv')

print("Размер набора данных:", df.shape)
print()

# Подготавливаем данные: используем Gr Liv Area как признак, SalePrice как целевую переменную
X = df[['Gr Liv Area']].values
y = df['SalePrice'].values

print(f"Размер X: {X.shape}")
print(f"Размер y: {y.shape}")
print()

# Создаем и обучаем RANSAC регрессор
# Используем LinearRegression как базовую модель
ransac = RANSACRegressor(
    LinearRegression(),
    max_trials=100,         # максимальное количество итераций
    min_samples=0.95,       # минимальное количество обучающих примеров (95%)
    residual_threshold=None,  # использует MAD (Median Absolute Deviation)
    random_state=123
)

ransac.fit(X, y)

# Получаем инлаеры и аутлаеры
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

print("=== РЕЗУЛЬТАТЫ RANSAC ===")
print(f"Количество инлаеров: {np.sum(inlier_mask)}")
print(f"Количество аутлаеров: {np.sum(outlier_mask)}")
print(f"Процент аутлаеров: {np.sum(outlier_mask) / len(y) * 100:.2f}%")
print()

# Выводим коэффициенты модели
print(f"Наклон (slope): {ransac.estimator_.coef_[0]:.3f}")
print(f"Пересечение (intercept): {ransac.estimator_.intercept_:.3f}")
print()

# Функция для расчета медианного абсолютного отклонения (MAD)
def median_absolute_deviation(data):
    return np.median(np.abs(data - np.median(data)))

# Расчет MAD для целевой переменной
mad = median_absolute_deviation(y)
print(f"MAD (Median Absolute Deviation) для SalePrice: {mad:,.2f}")
print()

# Визуализация RANSAC с residual_threshold=None (использует MAD)
line_X = np.arange(3, 10, 1)
line_y_ransac = ransac.predict(line_X[:, np.newaxis])

plt.scatter(X[inlier_mask], y[inlier_mask],
            c='steelblue', edgecolor='white',
            marker='o', label='Инлаеры')
plt.scatter(X[outlier_mask], y[outlier_mask],
            c='limegreen', edgecolor='white',
            marker='s', label='Аутлаеры')
plt.plot(line_X, line_y_ransac, color='black', lw=2)
plt.xlabel('Жилая площадь над землей, кв. футы')
plt.ylabel('Цена продажи, долл. США')
plt.legend(loc='upper left')
plt.title('RANSAC регрессия (residual_threshold=None, использует MAD)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ransac_regression_mad.png', dpi=300, bbox_inches='tight')
print("Сохранен график RANSAC (MAD): ransac_regression_mad.png")
plt.show()

# Пример с residual_threshold=65000 (меньшее количество аутлаеров)
print("=== RANSAC С residual_threshold=65000 ===")
ransac_custom = RANSACRegressor(
    LinearRegression(),
    max_trials=100,
    min_samples=0.95,
    residual_threshold=65000,  # пользовательский порог
    random_state=123
)

ransac_custom.fit(X, y)

inlier_mask_custom = ransac_custom.inlier_mask_
outlier_mask_custom = np.logical_not(inlier_mask_custom)

print(f"Количество инлаеров: {np.sum(inlier_mask_custom)}")
print(f"Количество аутлаеров: {np.sum(outlier_mask_custom)}")
print(f"Процент аутлаеров: {np.sum(outlier_mask_custom) / len(y) * 100:.2f}%")
print()

print(f"Наклон (slope): {ransac_custom.estimator_.coef_[0]:.3f}")
print(f"Пересечение (intercept): {ransac_custom.estimator_.intercept_:.3f}")
print()

# Визуализация RANSAC с residual_threshold=65000
line_y_ransac_custom = ransac_custom.predict(line_X[:, np.newaxis])

plt.scatter(X[inlier_mask_custom], y[inlier_mask_custom],
            c='steelblue', edgecolor='white',
            marker='o', label='Инлаеры')
plt.scatter(X[outlier_mask_custom], y[outlier_mask_custom],
            c='limegreen', edgecolor='white',
            marker='s', label='Аутлаеры')
plt.plot(line_X, line_y_ransac_custom, color='black', lw=2)
plt.xlabel('Жилая площадь над землей, кв. футы')
plt.ylabel('Цена продажи, долл. США')
plt.legend(loc='upper left')
plt.title('RANSAC регрессия (residual_threshold=65000)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ransac_regression_custom.png', dpi=300, bbox_inches='tight')
print("Сохранен график RANSAC (custom): ransac_regression_custom.png")
plt.show()

# Сравнение с обычной линейной регрессией
print("=== СРАВНЕНИЕ С ОБЫЧНОЙ ЛИНЕЙНОЙ РЕГРЕССИЕЙ ===")
slr = LinearRegression()
slr.fit(X, y)

print("Обычная линейная регрессия:")
print(f"  Наклон: {slr.coef_[0]:.3f}")
print(f"  Пересечение: {slr.intercept_:.3f}")
print()

print("RANSAC (MAD):")
print(f"  Наклон: {ransac.estimator_.coef_[0]:.3f}")
print(f"  Пересечение: {ransac.estimator_.intercept_:.3f}")
print()

print("RANSAC (residual_threshold=65000):")
print(f"  Наклон: {ransac_custom.estimator_.coef_[0]:.3f}")
print(f"  Пересечение: {ransac_custom.estimator_.intercept_:.3f}")
print()

print("RANSAC уменьшает влияние выбросов на коэффициенты модели.")
