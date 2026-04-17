import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ch2_3_2_adaline import AdalineGD
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

print("Тестирование: Setosa vs Virginica")
print("=" * 50)

# Выбираем Setosa и Virginica
# Setosa: строки 0-49
# Virginica: строки 100-149
y_setosa = df.iloc[0:50, 4].values
y_virginica = df.iloc[100:150, 4].values

# Объединяем данные
y = np.concatenate([y_setosa, y_virginica])
y = np.where(y == 'Iris-setosa', 0, 1)  # Setosa=0, Virginica=1

X_setosa = df.iloc[0:50, [0, 2]].values  # sepal_length, petal_length
X_virginica = df.iloc[100:150, [0, 2]].values
X = np.vstack([X_setosa, X_virginica])

print(f"Форма данных X: {X.shape}")
print(f"Количество образцов: {len(X)}")
print(f"Setosa: {np.sum(y == 0)} образцов")
print(f"Virginica: {np.sum(y == 1)} образцов")

# Стандартизация
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

print(f"\nДиапазон стандартизированных признаков:")
print(f"  Признак 1: [{X_std[:, 0].min():.2f}, {X_std[:, 0].max():.2f}]")
print(f"  Признак 2: [{X_std[:, 1].min():.2f}, {X_std[:, 1].max():.2f}]")

# Обучение Adaline
print(f"\nОбучение Adaline:")
print(f"Параметры: η=0.01, эпох=1000")

ada_gd = AdalineGD(n_iter=1000, eta=0.01)
ada_gd.fit(X_std, y)

print(f"Финальная ошибка: {ada_gd.losses_[-1]:.6f}")
print(f"Точность: {np.mean(ada_gd.predict(X_std) == y) * 100:.1f}%")
print(f"Веса: {ada_gd.w_}")
print(f"Смещение: {ada_gd.b_:.6f}")

# Анализ сходимости
convergence_epoch = None
for i, loss in enumerate(ada_gd.losses_):
    if loss < 0.01:
        convergence_epoch = i + 1
        break

if convergence_epoch:
    print(f"Сходимость достигнута на эпохе: {convergence_epoch}")
else:
    print("Сходимость не достигнута за 1000 эпох")

# Детальный анализ ошибок
predictions = ada_gd.predict(X_std)
misclassified = np.where(predictions != y)[0]

print(f"\nДетальный анализ ошибок:")
print(f"Всего ошибочно классифицировано: {len(misclassified)} из {len(y)}")

if len(misclassified) > 0:
    print(f"Индексы ошибочных образцов: {misclassified[:10]}")
    print(f"Ошибочные образцы (первые 3):")
    for i in misclassified[:3]:
        x1, x2 = X_std[i]
        net_input = ada_gd.w_[0] * x1 + ada_gd.w_[1] * x2 + ada_gd.b_
        true_class = y[i]
        pred_class = predictions[i]
        print(f"  Образец {i}: [{x1:.3f}, {x2:.3f}]")
        print(f"    net_input = {net_input:.3f}")
        print(f"    Истинный: {true_class}, Предсказанный: {pred_class}")
else:
    print("✅ Все образцы классифицированы правильно!")

# Точность по классам
print(f"\nТочность по классам:")
for cls in [0, 1]:
    mask = y == cls
    acc = np.mean(predictions[mask] == y[mask]) * 100
    class_name = "Setosa" if cls == 0 else "Virginica"
    print(f"{class_name} (Class {cls}): {acc:.1f}% ({np.sum(predictions[mask] == y[mask])}/{np.sum(mask)})")

# Визуализация
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plot_decision_regions(X_std, y, classifier=ada_gd)
plt.title('Adaline - Setosa vs Virginica')
plt.xlabel('Длина чашелистика [стандартизирована]')
plt.ylabel('Длина лепестка [стандартизирована]')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(range(1, len(ada_gd.losses_) + 1), ada_gd.losses_, marker='o')
plt.xlabel('Эпохи')
plt.ylabel('Среднеквадратичная ошибка')
plt.title('Кривая обучения Adaline')
plt.grid(True, alpha=0.3)

if convergence_epoch:
    plt.annotate(f'Сходимость\nна эпохе {convergence_epoch}', 
                xy=(convergence_epoch, ada_gd.losses_[convergence_epoch-1]),
                xytext=(convergence_epoch+50, ada_gd.losses_[0]*0.5),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green', ha='center')

plt.tight_layout()
plt.show()

print(f"\n💡 СРАВНЕНИЕ с Setosa vs Versicolor:")
print(f"Setosa vs Versicolor: 64.0% точность")
print(f"Setosa vs Virginica: {np.mean(ada_gd.predict(X_std) == y) * 100:.1f}% точность")

if np.mean(ada_gd.predict(X_std) == y) * 100 > 90:
    print("✅ Setosa vs Virginica - ЛУЧШАЯ комбинация!")
else:
    print("⚠️  Все еще есть проблемы с линейной разделимостью")
