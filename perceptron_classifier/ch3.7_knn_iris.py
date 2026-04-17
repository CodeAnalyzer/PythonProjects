"""
3.7. К-ближайшие соседи: "ленивый" алгоритм обучения

Последний алгоритм обучения с учителем, который мы хотим обсудить в этой главе, -
это классификатор по методу k-ближайших соседей (k-Nearest Neighbor, kNN). Он для
нас особенно интересен, поскольку принципиально отличается от алгоритмов обучения,
рассмотренных нами до сих пор.

kNN - типичный пример ленивого обучателя (lazy learner).

Сам алгоритм kNN довольно прост и может быть представлен в виде следующих шагов:
1. Выберите количество k и метрику расстояния.
2. Найдите k ближайших соседей записи данных, которую хотите классифицировать.
3. Присвойте текущей записи метку класса большинством голосов.

На основе выбранной метрики расстояния алгоритм kNN находит k экземпляров в обучающем
наборе данных, которые наиболее близки (наиболее похожи) к точке, которую
мы хотим классифицировать. Затем метку класса интересующей нас точки данных
определяют большинством голосов среди k ближайших соседей.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
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
print(f"Количество признаков: {X_train.shape[1]}")

# Стандартизация признаков
# Для kNN важно стандартизировать признаки, так как алгоритм основан на расстояниях
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

print(f"\nСредние значения до стандартизации (обучающая): {X_train.mean(axis=0)}")
print(f"Средние значения после стандартизации (обучающая): {X_train_std.mean(axis=0)}")
print(f"Стандартные отклонения после стандартизации (обучающая): {X_train_std.std(axis=0)}")

# Создание двух классификаторов k-ближайших соседей для сравнения
# n_neighbors=3 - количество соседей для голосования
# p=2 - параметр метрики Минковского (p=2 соответствует евклидову расстоянию)
# metric='minkowski' - метрика расстояния Минковского

# Классификатор с равными весами (uniform)
knn_uniform = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski', weights='uniform')
knn_uniform.fit(X_train_std, y_train)

# Классификатор с взвешиванием по расстоянию
knn_distance = KNeighborsClassifier(n_neighbors=3, p=2, metric='minkowski', weights='distance')
knn_distance.fit(X_train_std, y_train)

print(f"\n=== kNN с равными весами (uniform) ===")
print(f"Точность на обучающей выборке: {knn_uniform.score(X_train_std, y_train) * 100:.1f}%")
print(f"Точность на тестовой выборке: {knn_uniform.score(X_test_std, y_test) * 100:.1f}%")

print(f"\n=== kNN с взвешиванием по расстоянию (distance) ===")
print(f"Точность на обучающей выборке: {knn_distance.score(X_train_std, y_train) * 100:.1f}%")
print(f"Точность на тестовой выборке: {knn_distance.score(X_test_std, y_test) * 100:.1f}%")

# Объединение стандартизованных обучающих и тестовых данных для визуализации
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# Создание фигуры с двумя графиками рядом
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Первый график: равные веса (uniform)
plt.sca(axes[0])
plot_decision_regions(X_combined_std, y_combined,
                    classifier=knn_uniform, test_idx=range(105, 150))
plt.xlabel('Длина лепестка [стандартизирована]')
plt.ylabel('Ширина лепестка [стандартизирована]')
plt.title('kNN (k=3, weights=uniform)\nТочность на тесте: {:.1f}%'.format(knn_uniform.score(X_test_std, y_test) * 100))
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

# Второй график: взвешивание по расстоянию (distance)
plt.sca(axes[1])
plot_decision_regions(X_combined_std, y_combined,
                    classifier=knn_distance, test_idx=range(105, 150))
plt.xlabel('Длина лепестка [стандартизирована]')
plt.ylabel('Ширина лепестка [стандартизирована]')
plt.title('kNN (k=3, weights=distance)\nТочность на тесте: {:.1f}%'.format(knn_distance.score(X_test_std, y_test) * 100))
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Демонстрация работы алгоритма на конкретном примере
print(f"\nДемонстрация классификации нового примера (с взвешиванием по расстоянию):")
new_sample = np.array([[1.5, 0.5]])  # Пример: длина лепестка=1.5, ширина лепестка=0.5
new_sample_std = sc.transform(new_sample)
prediction = knn_distance.predict(new_sample_std)
distances, indices = knn_distance.kneighbors(new_sample_std)

print(f"Новый пример: длина лепестка=1.5 см, ширина лепестка=0.5 см")
class_names = ['Setosa', 'Versicolor', 'Virginica']
print(f"Предсказанный класс: {prediction[0]} ({class_names[prediction[0]]})")
print(f"Индексы {knn_distance.n_neighbors} ближайших соседей: {indices[0]}")
print(f"Расстояния до соседей: {distances[0].ravel()}")
neighbor_classes = [class_names[cls] for cls in y_train[indices[0]]]
print(f"Классы соседей: {neighbor_classes}")
