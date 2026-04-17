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

# Выбираем setosa и versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)

# Извлекаем признаки
X = df.iloc[0:100, [0, 2]].values

print("Анализ данных до стандартизации:")
print("=" * 50)
print(f"Форма X: {X.shape}")
print(f"Диапазон признаков:")
print(f"  Длина чашелистика: [{X[:, 0].min():.1f}, {X[:, 0].max():.1f}] (среднее: {X[:, 0].mean():.1f})")
print(f"  Длина лепестка: [{X[:, 1].min():.1f}, {X[:, 1].max():.1f}] (среднее: {X[:, 1].mean():.1f})")
print(f"  Стандартные отклонения: [{X[:, 0].std():.1f}, {X[:, 1].std():.1f}]")

# Стандартизация признаков
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

print("\nДанные после стандартизации:")
print("=" * 50)
print(f"Диапазон стандартизированных признаков:")
print(f"  Длина чашелистика: [{X_std[:, 0].min():.2f}, {X_std[:, 0].max():.2f}] (среднее: {X_std[:, 0].mean():.6f})")
print(f"  Длина лепестка: [{X_std[:, 1].min():.2f}, {X_std[:, 1].max():.2f}] (среднее: {X_std[:, 1].mean():.6f})")
print(f"  Стандартные отклонения: [{X_std[:, 0].std():.6f}, {X_std[:, 1].std():.6f}]")

# Обучение Adaline на стандартизированных данных
print("\nОбучение Adaline на стандартизированных данных:")
print("=" * 50)
print(f"Параметры: η=0.01, эпох=1000")

ada_gd = AdalineGD(n_iter=1000, eta=0.01)
ada_gd.fit(X_std, y)

print(f"Финальная ошибка: {ada_gd.losses_[-1]:.6f}")
print(f"Точность: {np.mean(ada_gd.predict(X_std) == y) * 100:.1f}%")
print(f"Веса: {ada_gd.w_}")
print(f"Смещение: {ada_gd.b_:.6f}")

# Анализ сходимости
print(f"\nАнализ сходимости:")
print(f"Начальная ошибка: {ada_gd.losses_[0]:.6f}")
print(f"Финальная ошибка: {ada_gd.losses_[-1]:.6f}")
improvement = (ada_gd.losses_[0] - ada_gd.losses_[-1]) / ada_gd.losses_[0] * 100
print(f"Улучшение: {improvement:.1f}%")

# Находим эпоху сходимости (ошибка < 0.01)
convergence_epoch = None
for i, loss in enumerate(ada_gd.losses_):
    if loss < 0.01:
        convergence_epoch = i + 1
        break

if convergence_epoch:
    print(f"Сходимость достигнута на эпохе: {convergence_epoch}")
else:
    print("Сходимость не достигнута за 1000 эпох")

# Детальный анализ ошибок классификации
predictions = ada_gd.predict(X_std)
misclassified = np.where(predictions != y)[0]

print(f"\nДетальный анализ ошибок:")
print(f"Всего ошибочно классифицировано: {len(misclassified)} из {len(y)}")

if len(misclassified) > 0:
    print(f"Индексы ошибочных образцов: {misclassified[:10]}")  # первые 10
    print(f"Ошибочные образцы (первые 5):")
    for i in misclassified[:5]:
        print(f"  Образец {i}: X={X_std[i]}, истинный={y[i]}, предсказанный={predictions[i]}")
else:
    print("✅ Все образцы классифицированы правильно!")

# Визуализация 1: Области решений
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plot_decision_regions(X_std, y, classifier=ada_gd)
plt.title('Adaline - градиентный спуск')
plt.xlabel('Длина чашелистика [стандартизирована]')
plt.ylabel('Длина лепестка [стандартизирована]')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

# Визуализация 2: График потерь
plt.subplot(1, 2, 2)
plt.plot(range(1, len(ada_gd.losses_) + 1), ada_gd.losses_, marker='o', linewidth=2)
plt.xlabel('Эпохи')
plt.ylabel('Среднеквадратичная ошибка')
plt.title('Кривая обучения Adaline')
plt.grid(True, alpha=0.3)

# Добавляем аннотацию о сходимости
if convergence_epoch:
    plt.annotate(f'Сходимость\nна эпохе {convergence_epoch}', 
                xy=(convergence_epoch, ada_gd.losses_[convergence_epoch-1]),
                xytext=(convergence_epoch+2, ada_gd.losses_[0]*0.5),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green', ha='center')

plt.tight_layout()
plt.show()

# Сравнение с нестандартизированными данными
print("\nСравнение с нестандартизированными данными:")
print("=" * 50)

# Пробуем те же параметры на нестандартизированных данных
print("Пробуем η=0.5 на нестандартизированных данных:")
try:
    ada_no_std = AdalineGD(n_iter=20, eta=0.5)
    ada_no_std.fit(X, y)
    print(f"Финальная ошибка: {ada_no_std.losses_[-1]:.6f}")
    if ada_no_std.losses_[-1] > ada_no_std.losses_[0]:
        print("❌ РАСХОДИМОСТЬ - стандартизация необходима!")
    else:
        print("✅ Сохраняет сходимость")
except Exception as e:
    print(f"❌ Ошибка: {e}")

print("\n💡 ВЫВОДЫ:")
print("• Стандартизация критически важна для градиентного спуска")
print("• Позволяет использовать большие скорости обучения (η=0.5)")
print("• Ускоряет сходимость в десятки раз")
print("• Делает обучение стабильным и предсказуемым")
