"""
3.6.2. Построение дерева решений

Деревья решений могут создавать сложные решающие границы, разделяя пространство
признаков на прямоугольники. Однако необходимо соблюдать осторожность, т. к. чем
глубже дерево решений, тем сложнее становится решающая граница, что может легко
привести к переобучению. Используя библиотеку scikit-learn, мы будем обучать дерево
решений с максимальной глубиной 4, используя в качестве критерия примесь Джини.
Хотя масштабирование признаков может пригодиться для более наглядной визуализации,
оно не является обязательным условием при использовании алгоритмов дерева решений.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
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
# Используем все три класса для многоклассовой классификации
X = df.iloc[:, [2, 3]].values  # petal_length и petal_width
y = df.iloc[:, 4].values

# Преобразование строковых меток классов в числовые
# Iris-setosa -> 0, Iris-versicolor -> 1, Iris-virginica -> 2
y = np.where(y == 'Iris-setosa', 0,
             np.where(y == 'Iris-versicolor', 1, 2))

# Разделение данных на обучающую и тестовую выборки
# Первые 105 образцов для обучения, последние 45 для тестирования
X_train, X_test = X[:105], X[105:]
y_train, y_test = y[:105], y[105:]

print(f"Размер обучающей выборки: {X_train.shape[0]} образцов")
print(f"Размер тестовой выборки: {X_test.shape[0]} образцов")
print(f"Количество классов: {len(np.unique(y))}")

# Создание и обучение дерева решений
# criterion='gini' - использование примеси Джини для оценки качества разбиения
# max_depth=4 - ограничение глубины дерева для предотвращения переобучения
# random_state=1 - фиксация случайного состояния для воспроизводимости результатов
tree_model = DecisionTreeClassifier(criterion='gini',
                                   max_depth=4,
                                   random_state=1)

# Обучение модели на обучающих данных
tree_model.fit(X_train, y_train)

print(f"\nГлубина дерева: {tree_model.get_depth()}")
print(f"Количество листьев: {tree_model.get_n_leaves()}")
print(f"Точность на обучающей выборке: {tree_model.score(X_train, y_train) * 100:.1f}%")
print(f"Точность на тестовой выборке: {tree_model.score(X_test, y_test) * 100:.1f}%")

# Объединение обучающих и тестовых данных для визуализации
X_combined = np.vstack((X_train, X_test))
y_combined = np.hstack((y_train, y_test))

# Создание графика для визуализации решающих границ
plt.figure(figsize=(10, 6))

# Визуализация областей решений
# test_idx=range(105, 150) - индексы тестовых образцов для выделения
plot_decision_regions(X_combined,
                    y_combined,
                    classifier=tree_model,
                    test_idx=range(105, 150))

# Настройка осей и заголовка
plt.xlabel('Длина лепестка [см]')
plt.ylabel('Ширина лепестка [см]')
plt.title('Дерево решений (max_depth=4) - Классификация Iris')
plt.legend(loc='upper left')
plt.tight_layout()
plt.grid(True, alpha=0.3)
plt.show()

# Вывод важности признаков
print(f"\nВажность признаков:")
feature_names = ['Длина лепестка', 'Ширина лепестка']
for name, importance in zip(feature_names, tree_model.feature_importances_):
    print(f"{name}: {importance:.3f}")

# Визуализация структуры дерева решений
# Библиотека scikit-learn позволяет легко визуализировать модель дерева решений
plt.figure(figsize=(12, 8))
tree.plot_tree(tree_model,
               feature_names=feature_names,
               filled=True,
               rounded=True,
               class_names=['Setosa', 'Versicolor', 'Virginica'])
plt.title('Структура дерева решений')
plt.show()
