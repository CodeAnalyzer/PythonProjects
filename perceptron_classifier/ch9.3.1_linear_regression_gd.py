"""
Глава 9.3.1: Нахождение регрессии для параметров регрессии с градиентным спуском

Реализация линейной регрессии с градиентным спуском (адаптация Adaline без пороговой функции).
Функция потерь - среднеквадратичная ошибка (MSE).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Загружаем очищенные данные
df = pd.read_csv('ames_housing_clean.csv')

print("Размер набора данных:", df.shape)
print()

# Класс линейной регрессии с градиентным спуском
class LinearRegressionGD:
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.array([0.])
        self.losses_ = []
        
        for i in range(self.n_iter):
            output = self.net_input(X)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (errors**2).mean()
            self.losses_.append(loss)
        return self
    
    def net_input(self, X):
        return np.dot(X, self.w_) + self.b_
    
    def predict(self, X):
        return self.net_input(X)

# Подготавливаем данные: используем Gr Liv Area как признак, SalePrice как целевую переменную
X = df[['Gr Liv Area']].values
y = df['SalePrice'].values

print(f"Размер X: {X.shape}")
print(f"Размер y: {y.shape}")
print()

# Стандартизируем переменные для лучшей сходимости GD
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:, np.newaxis]).flatten()

print("Данные стандартизированы")
print()

# Обучаем модель линейной регрессии с градиентным спуском
lr = LinearRegressionGD(eta=0.1)
lr.fit(X_std, y_std)

print("=== РЕЗУЛЬТАТЫ ОБУЧЕНИЯ ===")
print(f"Количество эпох: {lr.n_iter}")
print(f"Финальная потеря (MSE): {lr.losses_[-1]:.6f}")
print()

# Визуализация функции потерь
plt.plot(range(1, lr.n_iter + 1), lr.losses_)
plt.ylabel('MSE')
plt.xlabel('Epoch')
plt.title('Градиентный спуск: изменение MSE')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('linear_regression_gd_loss.png', dpi=300, bbox_inches='tight')
print("Сохранен график потерь: linear_regression_gd_loss.png")
plt.show()

# Вспомогательная функция для визуализации линии регрессии
def lin_regplot(X, y, model):
    plt.scatter(X, y, c='steelblue', edgecolor='white', s=70)
    plt.plot(X, model.predict(X), color='black', lw=2)
    plt.xlabel('Жилая площадь над землей (стандартизирована)')
    plt.ylabel('Цена продажи (стандартизирована)')
    plt.title('Линейная регрессия: Gr Liv Area vs SalePrice')
    plt.grid(True, alpha=0.3)

# Визуализация линии регрессии на стандартизированных данных
lin_regplot(X_std, y_std, lr)
plt.tight_layout()
plt.savefig('linear_regression_gd_fit.png', dpi=300, bbox_inches='tight')
print("Сохранен график регрессии: linear_regression_gd_fit.png")
plt.show()

# Прогноз цены дома с жилой площадью 2500 кв. футов
feature_std = sc_x.transform(np.array([[2500]]))
target_std = lr.predict(feature_std)
target_reverted = sc_y.inverse_transform(target_std.reshape(-1, 1))
print(f"Прогноз цены дома с площадью 2500 кв. футов: ${target_reverted.flatten()[0]:.2f}")
print()

# Вывод параметров модели
print(f"Slope (наклон): {lr.w_[0]:.3f}")
print(f"Intercept (смещение): {lr.b_[0]:.3f}")
print()

# Примечание: при работе со стандартизированными переменными
# точка пересечения (intercept) всегда равна 0
