"""
Глава 9.8.2: Регрессия на основе случайного леса

Случайный лес - ансамблевый метод, объединяющий несколько деревьев решений.
Для регрессии используется критерий MSE для выращивания деревьев,
а прогноз вычисляется как среднее по всем деревьям.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Загружаем очищенные данные
df = pd.read_csv('ames_housing_clean.csv')

print("Размер набора данных:", df.shape)
print()

# Используем все признаки для регрессии случайного леса
target = 'SalePrice'
features = df.columns[df.columns != target]

X = df[features].values
y = df[target].values

print("Используемые признаки:", features.tolist())
print(f"Размер X: {X.shape}")
print(f"Размер y: {y.shape}")
print()

# Разделяем данные на обучающий и тестовый наборы (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)

print(f"Размер обучающей выборки: {X_train.shape[0]}")
print(f"Размер тестовой выборки: {X_test.shape[0]}")
print()

# Создаем и обучаем модель случайного леса для регрессии
forest = RandomForestRegressor(
    n_estimators=1000,         # количество деревьев
    criterion='squared_error',  # критерий MSE
    random_state=1,
    n_jobs=-1                 # использовать все ядра процессора
)

print("Обучение модели случайного леса...")
forest.fit(X_train, y_train)
print("Обучение завершено")
print()

# Получаем предсказания
y_train_pred = forest.predict(X_train)
y_test_pred = forest.predict(X_test)

# Вычисляем MAE
mae_train = mean_absolute_error(y_train, y_train_pred)
mae_test = mean_absolute_error(y_test, y_test_pred)

print("=== СРЕДНЯЯ АБСОЛЮТНАЯ ОШИБКА (MAE) ===")
print(f'MAE при обучении: {mae_train:.2f}')
print(f'MAE при тестировании: {mae_test:.2f}')
print()

# Вычисляем R²
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print("=== КОЭФФИЦИЕНТ ОПРЕДЕЛЕНИЯ (R²) ===")
print(f'R² при обучении: {r2_train:.2f}')
print(f'R² при тестировании: {r2_test:.2f}')
print()

# Анализ результатов
print("=== АНАЛИЗ РЕЗУЛЬТАТОВ ===")
if r2_train > r2_test:
    print("Случайный лес склонен к переобучению на обучающих данных")
    print(f"Разрыв R²: {r2_train - r2_test:.2f}")
print()

print(f"Случайный лес объясняет {r2_test * 100:.1f}% дисперсии в тестовом наборе")
print()

# Сравнение с линейной моделью (из предыдущего раздела)
print("=== СРАВНЕНИЕ С ЛИНЕЙНОЙ МОДЕЛЬЮ ===")
print("Линейная модель (из разд. 9.5):")
print("  R² на тестовом наборе: ~0.75")
print("  Менее склонна к переобучению")
print()
print("Случайный лес:")
print(f"  R² на тестовом наборе: {r2_test:.2f}")
print("  Склонен к переобучению, но лучше работает на тестовом наборе")
print()

# Строим графики остатков
x_max = np.max([np.max(y_train_pred), np.max(y_test_pred)])
x_min = np.min([np.min(y_train_pred), np.min(y_test_pred)])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

ax1.scatter(y_test_pred, y_test_pred - y_test,
            c='limegreen', marker='s', edgecolor='white',
            label='Тестовые данные')
ax2.scatter(y_train_pred, y_train_pred - y_train,
            c='steelblue', marker='o', edgecolor='white',
            label='Обучающие данные')

ax1.set_ylabel('Остатки')

for ax in (ax1, ax2):
    ax.set_xlabel('Прогнозные значения')
    ax.legend(loc='upper left')
    ax.hlines(y=0, xmin=x_min-100, xmax=x_max+100,
              color='black', lw=2)
    ax.grid(True, alpha=0.3)

plt.suptitle('Графики остатков для случайного леса', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('random_forest_residual_plots.png', dpi=300, bbox_inches='tight')
print("Сохранены графики остатков: random_forest_residual_plots.png")
plt.show()

# Дополнительная визуализация: важность признаков
feature_importance = forest.feature_importances_
indices = np.argsort(feature_importance)[::-1]

plt.figure(figsize=(10, 6))
plt.title('Важность признаков в случайном лесе', fontsize=14, fontweight='bold')
plt.bar(range(len(features)), feature_importance[indices],
        color='steelblue', edgecolor='black', alpha=0.7)
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=45, ha='right')
plt.ylabel('Важность', fontsize=12)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('random_forest_feature_importance.png', dpi=300, bbox_inches='tight')
print("Сохранен график важности признаков: random_forest_feature_importance.png")
plt.show()

# Вывод важности признаков
print("=== ВАЖНОСТЬ ПРИЗНАКОВ ===")
for idx in indices:
    print(f"{features[idx]}: {feature_importance[idx]:.4f}")
