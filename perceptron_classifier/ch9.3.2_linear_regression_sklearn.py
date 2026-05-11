"""
Глава 9.3.2: Оценка коэффициента регрессионной модели с помощью scikit-learn

Реализация линейной регрессии с использованием LinearRegression из scikit-learn.
scikit-learn использует оптимизированную реализацию метода наименьших квадратов
из SciPy (scipy.linalg.lstsq), основанную на LAPACK.
Работает с нестандартизированными переменными.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Загружаем очищенные данные
df = pd.read_csv('ames_housing_clean.csv')

print("Размер набора данных:", df.shape)
print()

# Подготавливаем данные: используем Gr Liv Area как признак, SalePrice как целевую переменную
# Используем НЕСТАНДАРТИЗИРОВАННЫЕ переменные (в отличие от GD)
X = df[['Gr Liv Area']].values
y = df['SalePrice'].values

print(f"Размер X: {X.shape}")
print(f"Размер y: {y.shape}")
print()

# Создаем и обучаем модель линейной регрессии из scikit-learn
slr = LinearRegression()
slr.fit(X, y)

# Получаем предсказания
y_pred = slr.predict(X)

print("=== РЕЗУЛЬТАТЫ ОБУЧЕНИЯ ===")
print(f"Наклон (slope): {slr.coef_[0]:.3f}")
print(f"Пересечение (intercept): {slr.intercept_:.3f}")
print()

# Вспомогательная функция для визуализации линии регрессии
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)
    plt.xlabel('Жилая площадь над землей, кв. футы')
    plt.ylabel('Цена продажи, долл. США')
    plt.title('Линейная регрессия: Gr Liv Area vs SalePrice (scikit-learn)')
    plt.grid(True, alpha=0.3)

# Визуализация линии регрессии на нестандартизированных данных
lin_regplot(X, y, slr)
plt.tight_layout()
plt.savefig('linear_regression_sklearn_fit.png', dpi=300, bbox_inches='tight')
print("Сохранен график регрессии: linear_regression_sklearn_fit.png")
plt.show()

# Вычисляем метрики качества
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y, y_pred)

print("=== МЕТРИКИ КАЧЕСТВА ===")
print(f"MSE: {mse:,.2f}")
print(f"RMSE: {rmse:,.2f}")
print(f"R²: {r2:.4f}")
print()

# Прогноз цены дома с жилой площадью 2500 кв. футов
new_area = np.array([[2500]])
predicted_price = slr.predict(new_area)
print(f"Прогноз цены дома с площадью 2500 кв. футов: ${predicted_price[0]:.2f}")
print()

# Сравнение с реализацией GD
print("=== СРАВНЕНИЕ С РЕАЛИЗАЦИЕЙ GD ===")
print("scikit-learn использует нестандартизированные переменные,")
print("поэтому коэффициенты отличаются от реализации GD.")
print("Однако визуально результат должен быть идентичным.")
print()
print("Преимущества scikit-learn:")
print("  - Использует оптимизированную реализацию LAPACK")
print("  - Работает быстрее для больших наборов данных")
print("  - Не требует стандартизации переменных")
