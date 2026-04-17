import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from ch2_2_1_perceptron import Perceptron

def plot_decision_regions(X, y, classifier, resolution=0.02):
    """Визуализация решающих границ для двумерных данных"""
    
    # Настройка генератора меток и цветовой карты
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # Построение решающей поверхности
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    # Предсказание для всех точек сетки
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    
    # Отображение решающих границ
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # Построение образцов класса
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                   y=X[y == cl, 1],
                   alpha=0.8,
                   c=colors[idx],
                   marker=markers[idx],
                   label=f'Class {cl}',
                   edgecolor='black')

# Загрузка и подготовка данных
df = pd.read_csv('iris.data', header=None, encoding='utf-8')
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Выбираем setosa и versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)

# Извлекаем признаки: длина чашелистика и длина лепестка
X = df.iloc[0:100, [0, 2]].values

# Обучаем персептрон
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

print("Результаты обучения:")
print(f"Веса: {ppn.w_}")
print(f"Смещение: {ppn.b_}")
print(f"Ошибки по эпохам: {ppn.errors_}")

# Визуализация решающих границ
plt.figure(figsize=(12, 8))
plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('Длина чашелистика [см]')
plt.ylabel('Длина лепестка [см]')
plt.legend(loc='upper left')
plt.title('Области решений персептрона на данных Iris')
plt.grid(True, alpha=0.3)
plt.show()

# Анализ разделяющей линии
print(f"\nУравнение разделяющей линии:")
print(f"{ppn.w_[0]:.3f} * sepal_length + {ppn.w_[1]:.3f} * petal_length + {ppn.b_:.3f} = 0")
print(f"\nТочность на обучающих данных: {np.mean(ppn.predict(X) == y) * 100:.1f}%")

# Проверка граничных точек
print(f"\nПроверка граничных точек:")
test_points = np.array([[4.5, 1.5], [5.5, 2.5], [6.0, 3.0], [7.0, 4.0]])
for point in test_points:
    prediction = ppn.predict([point])
    class_name = "Setosa" if prediction[0] == 0 else "Versicolor"
    print(f"Точка {point} -> Класс {prediction[0]} ({class_name})")
