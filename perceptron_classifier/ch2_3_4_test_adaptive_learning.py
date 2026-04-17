import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ch2_3_4_adaline_sgd_adaptive import AdalineSGDAdaptive
from ch2_3_4_adaline_sgd import AdalineSGD
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    """Визуализация решающих границ для двумерных данных."""
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

print("СРАВНЕНИЕ: ФИКСИРОВАННАЯ vs АДАПТИВНАЯ СКОРОСТЬ ОБУЧЕНИЯ")
print("=" * 70)

# Подготовка данных (Setosa vs Versicolor - сложный случай)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)
X = df.iloc[0:100, [0, 2]].values

# Стандартизация
X_std = np.copy(X)
X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()

print(f"Набор данных: Setosa vs Versicolor (100 образцов)")
print(f"Параметры: начальная eta=0.01, эпох=20")

# Тест 1: Фиксированная скорость обучения
print(f"\n🎯 ТЕСТ 1: ФИКСИРОВАННАЯ СКОРОСТЬ ОБУЧЕНИЯ")
print("-" * 50)

ada_fixed = AdalineSGD(n_iter=50, eta=0.01, random_state=1)
ada_fixed.fit(X_std, y)

accuracy_fixed = np.mean(ada_fixed.predict(X_std) == y) * 100
final_loss_fixed = ada_fixed.losses_[-1]

print(f"Финальная потеря: {final_loss_fixed:.6f}")
print(f"Точность: {accuracy_fixed:.1f}%")
print(f"Финальные веса: {ada_fixed.w_}")
print(f"Финальное смещение: {ada_fixed.b_:.6f}")

# Тест 2: Адаптивная скорость обучения
print(f"\n🎯 ТЕСТ 2: АДАПТИВНАЯ СКОРОСТЬ ОБУЧЕНИЯ")
print("-" * 50)

ada_adaptive = AdalineSGDAdaptive(n_iter=50, eta0=0.1, random_state=1, 
                                   adaptive=True, c1=5000.0, c2=100.0)
ada_adaptive.fit(X_std, y)

accuracy_adaptive = np.mean(ada_adaptive.predict(X_std) == y) * 100
final_loss_adaptive = ada_adaptive.losses_[-1]

print(f"Финальная потеря: {final_loss_adaptive:.6f}")
print(f"Точность: {accuracy_adaptive:.1f}%")
print(f"Финальные веса: {ada_adaptive.w_}")
print(f"Финальное смещение: {ada_adaptive.b_:.6f}")

# Анализ скорости обучения
print(f"\n📊 АНАЛИЗ СКОРОСТИ ОБУЧЕНИЯ")
print("-" * 50)

print(f"Фиксированная eta: {ada_fixed.eta:.6f} (постоянная)")
print(f"Адаптивная eta:")
print(f"  Начальная: {ada_adaptive.eta_history_[0]:.6f}")
print(f"  Финальная: {ada_adaptive.eta_history_[-1]:.6f}")
print(f"  Уменьшение в {ada_adaptive.eta_history_[0]/ada_adaptive.eta_history_[-1]:.1f} раз")

# Сравнение результатов
print(f"\n🏆 СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("-" * 50)

improvement_loss = ((final_loss_fixed - final_loss_adaptive) / final_loss_fixed) * 100
improvement_acc = accuracy_adaptive - accuracy_fixed

print(f"Потеря: {final_loss_fixed:.6f} → {final_loss_adaptive:.6f} "
      f"({improvement_loss:+.1f}%)")
print(f"Точность: {accuracy_fixed:.1f}% → {accuracy_adaptive:.1f}% "
      f"({improvement_acc:+.1f}%)")

if improvement_loss > 0:
    print("✅ Адаптивная скорость обучения показала лучшую сходимость!")
else:
    print("⚠️  Фиксированная скорость обучения показала лучший результат")

# Визуализация
plt.figure(figsize=(18, 12))

# График 1: Кривые обучения
plt.subplot(2, 4, 1)
plt.plot(range(1, len(ada_fixed.losses_) + 1), ada_fixed.losses_, 
         marker='o', label='Фиксированная eta', linewidth=2)
plt.plot(range(1, len(ada_adaptive.losses_) + 1), ada_adaptive.losses_, 
         marker='s', label='Адаптивная eta', linewidth=2)
plt.xlabel('Эпохи')
plt.ylabel('Средняя потеря')
plt.title('Кривые обучения')
plt.legend()
plt.grid(True, alpha=0.3)

# График 2: Изменение скорости обучения
plt.subplot(2, 4, 2)
plt.plot(range(1, len(ada_adaptive.eta_history_) + 1), ada_adaptive.eta_history_, 
         marker='s', color='red', linewidth=2)
plt.xlabel('Эпохи')
plt.ylabel('Скорость обучения (eta)')
plt.title('Адаптивная скорость обучения')
plt.grid(True, alpha=0.3)

# График 3: Области решений (фиксированная)
plt.subplot(2, 4, 3)
plot_decision_regions(X_std, y, classifier=ada_fixed)
plt.title('Фиксированная eta')
plt.xlabel('Длина чашелистика [стд.]')
plt.ylabel('Длина лепестка [стд.]')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

# График 4: Области решений (адаптивная)
plt.subplot(2, 4, 4)
plot_decision_regions(X_std, y, classifier=ada_adaptive)
plt.title('Адаптивная eta')
plt.xlabel('Длина чашелистика [стд.]')
plt.ylabel('Длина лепестка [стд.]')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

# График 5: Детализация кривых обучения (первые 10 эпох)
plt.subplot(2, 4, 5)
plt.plot(range(1, min(11, len(ada_fixed.losses_) + 1)), 
         ada_fixed.losses_[:10], marker='o', label='Фиксированная eta', linewidth=2)
plt.plot(range(1, min(11, len(ada_adaptive.losses_) + 1)), 
         ada_adaptive.losses_[:10], marker='s', label='Адаптивная eta', linewidth=2)
plt.xlabel('Эпохи')
plt.ylabel('Средняя потеря')
plt.title('Кривые обучения (первые 10 эпох)')
plt.legend()
plt.grid(True, alpha=0.3)

# График 6: Логарифмическая шкала скорости обучения
plt.subplot(2, 4, 6)
plt.semilogy(range(1, len(ada_adaptive.eta_history_) + 1), ada_adaptive.eta_history_, 
             marker='s', color='red', linewidth=2)
plt.xlabel('Эпохи')
plt.ylabel('Скорость обучения (log scale)')
plt.title('Адаптивная eta (логарифмическая шкала)')
plt.grid(True, alpha=0.3)

# График 7: Сравнение точности
plt.subplot(2, 4, 7)
accuracies = [accuracy_fixed, accuracy_adaptive]
labels = ['Фиксированная eta', 'Адаптивная eta']
colors = ['blue', 'red']

bars = plt.bar(labels, accuracies, color=colors, alpha=0.7)
plt.ylabel('Точность (%)')
plt.title('Сравнение точности')
plt.ylim(0, 100)

for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{acc:.1f}%', ha='center', va='bottom')

plt.grid(True, alpha=0.3, axis='y')

# График 8: Сравнение финальных потерь
plt.subplot(2, 4, 8)
losses = [final_loss_fixed, final_loss_adaptive]

bars = plt.bar(labels, losses, color=colors, alpha=0.7)
plt.ylabel('Финальная потеря')
plt.title('Сравнение финальных потерь')

for bar, loss in zip(bars, losses):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + loss*0.05, 
             f'{loss:.4f}', ha='center', va='bottom')

plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

print(f"\n💡 ТЕОРЕТИЧЕСКОЕ ОБОСНОВАНИЕ:")
print("=" * 70)
print("Адаптивная скорость обучения: eta(t) = c1 / (t + c2)")
print("• Начало обучения: высокая eta → быстрые шаги")
print("• Конец обучения: низкая eta → точная настройка")
print("• Преимущество: избегает 'перелетов' через минимум")
print("• Результат: лучшее приближение к глобальному минимуму")

print(f"\n🎯 ПРАКТИЧЕСКИЕ ВЫВОДЫ:")
print("=" * 70)
print("• Адаптивная eta помогает на сложных, нелинейно разделимых данных")
print("• Особенно полезна при большом количестве итераций")
print("• Требует настройки параметров c1 и c2")
print("• Комбинирует преимущества быстрого и точного обучения")
