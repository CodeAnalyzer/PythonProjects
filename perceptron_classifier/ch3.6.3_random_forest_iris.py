"""
Случайный лес (Random Forest)

Случайный лес - это ансамбль деревьев решений, который использует два вида рандомизации:
1. Бутстрэп-выборка (bootstrap sampling) - случайный выбор обучающих примеров с заменой
2. Случайный выбор признаков при каждом ветвлении

В большинстве реализаций, включая RandomForestClassifier в scikit-learn, размер бутстрэп-
выборки выбирают равным количеству обучающих экземпляров в исходном наборе
обучающих данных, что обычно обеспечивает хороший компромисс смещения и
дисперсии. Количество признаков d при каждом ветвлении должно быть меньше, чем
общее количество признаков в обучающем наборе данных. Разумным значением по
умолчанию, используемым в scikit-learn и других реализациях, является d = √m, где
m - количество признаков в обучающем наборе данных.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    """
    Визуализация областей решений с выделением тестовых образцов
    
    Параметры:
    -----------
    X : array-like
        Матрица признаков (n_samples, n_features)
    y : array-like
        Вектор меток классов (n_samples,)
    classifier : object
        Обученный классификатор с методом predict
    test_idx : array-like, optional
        Индексы тестовых образцов для выделения на графике
    resolution : float
        Разрешение сетки для визуализации решающих границ
    """
    # Настройка генератора маркеров и цветовой карты
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # Построение решающей поверхности
    # Определение границ области признаков
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Создание сетки точек для предсказания
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    # Предсказание класса для каждой точки сетки
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    # Отображение решающих границ с помощью контурного графика
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # Отображение всех образцов классов
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                   y=X[y == cl, 1],
                   alpha=0.8,
                   c=colors[idx],
                   marker=markers[idx],
                   label=f'Класс {cl}',
                   edgecolor='black')
    
    # Выделение тестовых образцов кружком
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                   c='none', edgecolor='black', alpha=1.0,
                   linewidth=1, marker='o',
                   s=100, label='Тестовые образцы')


# Загрузка набора данных Iris
df = pd.read_csv('iris.data', header=None, encoding='utf-8')
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Выбор только двух признаков: длина и ширина лепестка
# Используем только два класса (setosa и versicolor) для бинарной классификации
X = df.iloc[:100, [2, 3]].values  # petal_length и petal_width (первые 100 образцов)
y = df.iloc[:100, 4].values

# Преобразование строковых меток классов в числовые
# Iris-setosa -> 0, Iris-versicolor -> 1
y = np.where(y == 'Iris-setosa', 0, 1)

# Разделение данных на обучающую и тестовую выборки
# Первые 70 образцов для обучения, последние 30 для тестирования
X_train, X_test = X[:70], X[70:]
y_train, y_test = y[:70], y[70:]

print(f"Размер обучающей выборки: {X_train.shape[0]} образцов")
print(f"Размер тестовой выборки: {X_test.shape[0]} образцов")
print(f"Количество классов: {len(np.unique(y))}")
print(f"Количество признаков: {X_train.shape[1]}")

# Создание классификатора случайного леса
# n_estimators=25 - количество деревьев в лесу
# random_state=1 - фиксация случайного состояния для воспроизводимости
# n_jobs=2 - использование 2 ядер для параллельного обучения
forest = RandomForestClassifier(n_estimators=25,
                               random_state=1,
                               n_jobs=2)

# Обучение модели на обучающих данных
forest.fit(X_train, y_train)

print(f"\nТочность на обучающей выборке: {forest.score(X_train, y_train) * 100:.1f}%")
print(f"Точность на тестовой выборке: {forest.score(X_test, y_test) * 100:.1f}%")

# Объединение обучающих и тестовых данных для визуализации
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

# Создание графика для визуализации решающих границ
plt.figure(figsize=(10, 6))

# Визуализация областей решений, сформированных ансамблем деревьев в случайном лесу
# test_idx=range(70, 100) - индексы тестовых образцов для выделения
plot_decision_regions(X_combined, y_combined,
                    classifier=forest, test_idx=range(70, 100))

# Настройка осей и заголовка
plt.xlabel('Длина лепестка [см]')
plt.ylabel('Ширина лепестка [см]')
plt.title('Случайный лес (25 деревьев) - Классификация Iris (2 класса)')
plt.legend(loc='upper left')
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.show()

# Вывод важности признаков
print(f"\nВажность признаков:")
feature_names = ['Длина лепестка', 'Ширина лепестка']
for name, importance in zip(feature_names, forest.feature_importances_):
    print(f"{name}: {importance:.3f}")

# Вывод информации о деревьях в лесу
print(f"\nИнформация о случайном лесу:")
print(f"Количество деревьев: {forest.n_estimators}")
print(f"Количество признаков при разбиении: {forest.max_features}")
