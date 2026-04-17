import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ch2_2_1_perceptron import Perceptron
from ch2_3_2_adaline import AdalineGD

# Загрузка и подготовка данных
df = pd.read_csv('iris.data', header=None, encoding='utf-8')
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Выбираем setosa и versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)

# Извлекаем признаки
X = df.iloc[0:100, [0, 2]].values

# Стандартизация признаков для Adaline
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

print("Сравнение Perceptron vs AdalineGD")
print("=" * 50)

# 1. Perceptron
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

print("\nPerceptron:")
print(f"Веса: {ppn.w_}")
print(f"Смещение: {ppn.b_}")
print(f"Ошибки по эпохам: {ppn.errors_}")
print(f"Точность: {np.mean(ppn.predict(X) == y) * 100:.1f}%")

# 2. Adaline с разной скоростью обучения
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Adaline с eta=0.0001 (правильная скорость)
ada1 = AdalineGD(n_iter=1000, eta=0.0001)
ada1.fit(X_std, y)
axes[0, 0].plot(range(1, len(ada1.losses_) + 1), ada1.losses_, marker='o')
axes[0, 0].set_xlabel('Эпохи')
axes[0, 0].set_ylabel('Среднеквадратичная ошибка')
axes[0, 0].set_title('Adaline - Скорость обучения: 0.0001')
axes[0, 0].grid(True)

# Adaline с eta=0.00001 (очень медленная)
ada2 = AdalineGD(n_iter=100, eta=0.00001)
ada2.fit(X_std, y)
axes[0, 1].plot(range(1, len(ada2.losses_) + 1), ada2.losses_, marker='o')
axes[0, 1].set_xlabel('Эпохи')
axes[0, 1].set_ylabel('Среднеквадратичная ошибка')
axes[0, 1].set_title('Adaline - Скорость обучения: 0.00001')
axes[0, 1].grid(True)

# График ошибок персептрона
axes[1, 0].plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o', color='red')
axes[1, 0].set_xlabel('Эпохи')
axes[1, 0].set_ylabel('Количество ошибок')
axes[1, 0].set_title('Perceptron - Количество ошибок')
axes[1, 0].grid(True)

# Визуализация решающих границ Adaline
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    
    axes[1, 1].contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    axes[1, 1].set_xlim(xx1.min(), xx1.max())
    axes[1, 1].set_ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        axes[1, 1].scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                          alpha=0.8, c=colors[idx], marker=markers[idx],
                          label=f'Class {cl}', edgecolor='black')

plot_decision_regions(X_std, y, classifier=ada1)
axes[1, 1].set_xlabel('Стандартизированная длина чашелистика')
axes[1, 1].set_ylabel('Стандартизированная длина лепестка')
axes[1, 1].set_title('Области решений Adaline (η=0.0001)')
axes[1, 1].legend(loc='upper left')

plt.tight_layout()
plt.show()

# Сравнение результатов
print(f"\nAdaline (η=0.0001):")
print(f"Финальная ошибка: {ada1.losses_[-1]:.6f}")
print(f"Точность: {np.mean(ada1.predict(X_std) == y) * 100:.1f}%")

print(f"\nAdaline (η=0.00001):")
print(f"Финальная ошибка: {ada2.losses_[-1]:.6f}")
print(f"Точность: {np.mean(ada2.predict(X_std) == y) * 100:.1f}%")

print("\nКлючевые различия:")
print("• Perceptron: обновляет веса после каждого ошибочного примера")
print("• Adaline: обновляет веса на основе градиента функции потерь")
print("• Perceptron: использует пороговую функцию активации")
print("• Adaline: использует линейную активацию при обучении")
