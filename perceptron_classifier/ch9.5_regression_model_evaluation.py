"""
Глава 9.5: Оценка производительности моделей линейной регрессии

Оценка обобщающей способности регрессионной модели с использованием:
- Разделения на обучающий и тестовый наборы
- Графиков остатков
- Метрик MSE, MAE и R²
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Загружаем очищенные данные
df = pd.read_csv('ames_housing_clean.csv')

print("Размер набора данных:", df.shape)
print()

# Используем все пять признаков для множественной регрессии
target = 'SalePrice'
features = df.columns[df.columns != target]

X = df[features].values
y = df[target].values

print("Используемые признаки:", features.tolist())
print(f"Размер X: {X.shape}")
print(f"Размер y: {y.shape}")
print()

# Разделяем данные на обучающий и тестовый наборы
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)

print(f"Размер обучающей выборки: {X_train.shape[0]}")
print(f"Размер тестовой выборки: {X_test.shape[0]}")
print()

# Обучаем модель линейной регрессии
slr = LinearRegression()
slr.fit(X_train, y_train)

# Получаем предсказания
y_train_pred = slr.predict(X_train)
y_test_pred = slr.predict(X_test)

print("=== РЕЗУЛЬТАТЫ ОБУЧЕНИЯ ===")
print(f"Наклоны (коэффициенты): {slr.coef_}")
print(f"Пересечение (intercept): {slr.intercept_:.3f}")
print()

# Строим графики остатков
x_max = np.max([np.max(y_train_pred), np.max(y_test_pred)])
x_min = np.min([np.min(y_train_pred), np.min(y_test_pred)])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

ax1.scatter(y_test_pred, y_test_pred - y_test,
            c='limegreen', marker='s',
            edgecolor='white',
            label='Тестовые данные')
ax2.scatter(y_train_pred, y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Обучающие данные')

ax1.set_ylabel('Остатки')

for ax in (ax1, ax2):
    ax.set_xlabel('Предсказанные значения')
    ax.legend(loc='upper left')
    ax.hlines(y=0, xmin=x_min-100, xmax=x_max+100,
              color='black', lw=2)
    ax.grid(True, alpha=0.3)

plt.suptitle('Графики остатков для диагностики регрессионной модели', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('regression_residual_plots.png', dpi=300, bbox_inches='tight')
print("Сохранены графики остатков: regression_residual_plots.png")
plt.show()

# Вычисляем MSE
mse_train = mean_squared_error(y_train, y_train_pred)
mse_test = mean_squared_error(y_test, y_test_pred)

print("=== СРЕДНЕКВАДРАТИЧЕСКАЯ ОШИБКА (MSE) ===")
print(f'MSE при обучении: {mse_train:.2f}')
print(f'MSE при тестировании: {mse_test:.2f}')
print()

# Вычисляем MAE
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print("=== СРЕДНЯЯ АБСОЛЮТНАЯ ОШИБКА (MAE) ===")
print(f'MAE при обучении: {mae_train:.2f}')
print(f'MAE при тестировании: {mae_test:.2f}')
print()

print(f"Интерпретация: модель ошибается в среднем примерно на ${mae_test:,.0f}")
print()

# Вычисляем R²
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("=== КОЭФФИЦИЕНТ ОПРЕДЕЛЕНИЯ (R²) ===")
print(f'R² на обучающем наборе: {train_r2:.3f}')
print(f'R² на тестовом наборе: {test_r2:.3f}')
print()

# Анализ результатов
print("=== АНАЛИЗ РЕЗУЛЬТАТОВ ===")
if mse_train < mse_test:
    print("MSE на обучающем наборе меньше, чем на тестовом - возможное переобучение")
else:
    print("MSE на обучающем наборе больше или равно тестовому - модель хорошо обобщает")
print()

if train_r2 > test_r2:
    print("R² на обучающем наборе выше, чем на тестовом - возможное переобучение")
else:
    print("R² на обучающем наборе ниже или равно тестовому - модель хорошо обобщает")
print()

print("Качество модели:")
if test_r2 >= 0.8:
    print("  Отличное (R² >= 0.8)")
elif test_r2 >= 0.6:
    print("  Хорошее (0.6 <= R² < 0.8)")
elif test_r2 >= 0.4:
    print("  Удовлетворительное (0.4 <= R² < 0.6)")
else:
    print("  Плохое (R² < 0.4)")
print()

# Дополнительная визуализация: предсказанные vs реальные значения
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Обучающая выборка
ax1.scatter(y_train, y_train_pred, c='steelblue', alpha=0.5, edgecolor='white')
ax1.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 
         'k--', lw=2)
ax1.set_xlabel('Реальные значения')
ax1.set_ylabel('Предсказанные значения')
ax1.set_title(f'Обучающая выборка (R² = {train_r2:.3f})')
ax1.grid(True, alpha=0.3)

# Тестовая выборка
ax2.scatter(y_test, y_test_pred, c='limegreen', alpha=0.5, edgecolor='white')
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'k--', lw=2)
ax2.set_xlabel('Реальные значения')
ax2.set_ylabel('Предсказанные значения')
ax2.set_title(f'Тестовая выборка (R² = {test_r2:.3f})')
ax2.grid(True, alpha=0.3)

plt.suptitle('Предсказанные vs реальные значения', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('regression_predictions.png', dpi=300, bbox_inches='tight')
print("Сохранен график предсказаний: regression_predictions.png")
plt.show()
