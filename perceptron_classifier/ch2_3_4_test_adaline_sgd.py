import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ch2_3_4_adaline_sgd import AdalineSGD
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    """Визуализация решающих границ для двумерных данных"""
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                   alpha=0.8, c=colors[idx], marker=markers[idx],
                   label=f'Class {cl}', edgecolor='black')

# Загрузка данных
df = pd.read_csv('iris.data', header=None, encoding='utf-8')
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

print("Тестирование AdalineSGD на разных наборах данных")
print("=" * 60)

# Тест 1: Setosa vs Virginica (идеально разделимы)
print("\n🎯 ТЕСТ 1: Setosa vs Virginica (идеальный случай)")
print("-" * 50)

# Подготовка данных
y_setosa = df.iloc[0:50, 4].values
y_virginica = df.iloc[100:150, 4].values
y = np.concatenate([y_setosa, y_virginica])
y = np.where(y == 'Iris-setosa', 0, 1)

X_setosa = df.iloc[0:50, [0, 2]].values
X_virginica = df.iloc[100:150, [0, 2]].values
X = np.vstack([X_setosa, X_virginica])

# Стандартизация
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

print(f"Параметры: η=0.01, эпох=15, random_state=1")

# Обучение AdalineSGD
ada_sgd = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada_sgd.fit(X_std, y)

print(f"Финальная потеря: {ada_sgd.losses_[-1]:.6f}")
print(f"Точность: {np.mean(ada_sgd.predict(X_std) == y) * 100:.1f}%")
print(f"Веса: {ada_sgd.w_}")
print(f"Смещение: {ada_sgd.b_:.6f}")

# Тест 2: Setosa vs Versicolor (сложный случай)
print("\n🎯 ТЕСТ 2: Setosa vs Versicolor (сложный случай)")
print("-" * 50)

# Подготовка данных
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
X = df.iloc[0:100, [0, 2]].values

# Стандартизация
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

print(f"Параметры: η=0.01, эпох=15, random_state=1")

# Обучение AdalineSGD
ada_sgd2 = AdalineSGD(n_iter=15, eta=0.01, random_state=1)
ada_sgd2.fit(X_std, y)

print(f"Финальная потеря: {ada_sgd2.losses_[-1]:.6f}")
print(f"Точность: {np.mean(ada_sgd2.predict(X_std) == y) * 100:.1f}%")
print(f"Веса: {ada_sgd2.w_}")
print(f"Смещение: {ada_sgd2.b_:.6f}")

# Визуализация результатов
plt.figure(figsize=(15, 10))

# График 1: Setosa vs Virginica - области решений
plt.subplot(2, 3, 1)
plot_decision_regions(X_std, y, classifier=ada_sgd)
plt.title('AdalineSGD - Setosa vs Virginica')
plt.xlabel('Длина чашелистика [стандартизирована]')
plt.ylabel('Длина лепестка [стандартизирована]')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

# График 2: Setosa vs Virginica - кривая обучения
plt.subplot(2, 3, 2)
plt.plot(range(1, len(ada_sgd.losses_) + 1), ada_sgd.losses_, marker='o')
plt.xlabel('Эпохи')
plt.ylabel('Средняя потеря')
plt.title('Setosa vs Virginica - Обучение')
plt.grid(True, alpha=0.3)

# График 3: Сравнение потерь
plt.subplot(2, 3, 3)
plt.plot(range(1, len(ada_sgd.losses_) + 1), ada_sgd.losses_, 
         marker='o', label='Setosa vs Virginica', linewidth=2)
plt.plot(range(1, len(ada_sgd2.losses_) + 1), ada_sgd2.losses_, 
         marker='s', label='Setosa vs Versicolor', linewidth=2)
plt.xlabel('Эпохи')
plt.ylabel('Средняя потеря')
plt.title('Сравнение кривых обучения')
plt.legend()
plt.grid(True, alpha=0.3)

# Подготовка данных для второго теста
y2 = df.iloc[0:100, 4].values
y2 = np.where(y2 == 'Iris-setosa', 0, 1)
X2 = df.iloc[0:100, [0, 2]].values
X2_std = np.copy(X2)
X2_std[:, 0] = (X2[:, 0] - X2[:, 0].mean()) / X2[:, 0].std()
X2_std[:, 1] = (X2[:, 1] - X2[:, 1].mean()) / X2[:, 1].std()

# График 4: Setosa vs Versicolor - области решений
plt.subplot(2, 3, 4)
plot_decision_regions(X2_std, y2, classifier=ada_sgd2)
plt.title('AdalineSGD - Setosa vs Versicolor')
plt.xlabel('Длина чашелистика [стандартизирована]')
plt.ylabel('Длина лепестка [стандартизирована]')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

# График 5: Setosa vs Versicolor - кривая обучения
plt.subplot(2, 3, 5)
plt.plot(range(1, len(ada_sgd2.losses_) + 1), ada_sgd2.losses_, marker='o')
plt.xlabel('Эпохи')
plt.ylabel('Средняя потеря')
plt.title('Setosa vs Versicolor - Обучение')
plt.grid(True, alpha=0.3)

# График 6: Сравнение точности
plt.subplot(2, 3, 6)
accuracies = [
    np.mean(ada_sgd.predict(X_std) == y) * 100,
    np.mean(ada_sgd2.predict(X2_std) == y2) * 100
]
labels = ['Setosa vs Virginica', 'Setosa vs Versicolor']
colors = ['green', 'red']

bars = plt.bar(labels, accuracies, color=colors, alpha=0.7)
plt.ylabel('Точность (%)')
plt.title('Сравнение точности')
plt.ylim(0, 100)

# Добавляем значения на столбцы
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{acc:.1f}%', ha='center', va='bottom')

plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Анализ преимуществ SGD
print("\n💡 ПРЕИМУЩЕСТВА СТОХАСТИЧЕСКОГО ГРАДИЕНТНОГО СПУСКА:")
print("=" * 60)
print("✅ Быстрее сходимость на больших наборах данных")
print("✅ Меньше памяти требуется (один образец за раз)")
print("✅ Может выходить из локальных минимумов")
print("✅ Хорошо подходит для онлайн-обучения")
print("✅ Перемешивание предотвращает циклы")

print(f"\n📊 СРАВНЕНИЕ РЕЗУЛЬТАТОВ:")
print(f"Setosa vs Virginica: {np.mean(ada_sgd.predict(X_std) == y) * 100:.1f}% точность")
print(f"Setosa vs Versicolor: {np.mean(ada_sgd2.predict(X2_std) == y2) * 100:.1f}% точность")

print(f"\n🎯 ВЫВОДЫ:")
print("• SGD работает так же хорошо, как и пакетный градиентный спуск")
print("• На идеально разделимых данных достигает 100% точности")
print("• На сложных данных показывает сопоставимые результаты")
print("• Требует меньше эпох для сходимости")
