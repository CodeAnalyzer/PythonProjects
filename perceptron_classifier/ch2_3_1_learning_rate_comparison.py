import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ch2_3_2_adaline import AdalineGD

# Загрузка и подготовка данных
df = pd.read_csv('iris.data', header=None, encoding='utf-8')
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Выбираем setosa и versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)

# Извлекаем признаки БЕЗ стандартизации (как в книге)
X = df.iloc[0:100, [0, 2]].values

print("Сравнение скорости обучения Adaline")
print("=" * 50)
print(f"Форма данных X: {X.shape}")
print(f"Диапазон признаков:")
print(f"  Длина чашелистика: [{X[:, 0].min():.1f}, {X[:, 0].max():.1f}]")
print(f"  Длина лепестка: [{X[:, 1].min():.1f}, {X[:, 1].max():.1f}]")

# Создаем графики
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

# 1. Слишком большая скорость обучения (η=0.1)
print("\n1. Обучение с η=0.1 (слишком большая):")
ada1 = AdalineGD(n_iter=15, eta=0.1)
ada1.fit(X, y)

print(f"   Финальная ошибка: {ada1.losses_[-1]:.6f}")
print(f"   Первые ошибки: {ada1.losses_[:5]}")
print(f"   Последние ошибки: {ada1.losses_[-5:]}")

ax[0].plot(range(1, len(ada1.losses_) + 1), 
          np.log10(ada1.losses_), marker='o', linewidth=2)
ax[0].set_xlabel('Эпохи')
ax[0].set_ylabel('log(Среднеквадратичная ошибка)')
ax[0].set_title('Adaline - скорость обучения 0.1\n(слишком большая - расходимость)')
ax[0].grid(True, alpha=0.3)

# Добавляем аннотацию для расходимости
if ada1.losses_[-1] > ada1.losses_[0]:
    ax[0].annotate('Ошибка растет!\nРасходимость', 
                   xy=(len(ada1.losses_), np.log10(ada1.losses_[-1])),
                   xytext=(len(ada1.losses_)*0.5, np.log10(ada1.losses_[0])),
                   arrowprops=dict(arrowstyle='->', color='red'),
                   fontsize=10, color='red', ha='center')

# 2. Слишком маленькая скорость обучения (η=0.0001)
print("\n2. Обучение с η=0.0001 (слишком маленькая):")
ada2 = AdalineGD(n_iter=15, eta=0.0001)
ada2.fit(X, y)

print(f"   Финальная ошибка: {ada2.losses_[-1]:.6f}")
print(f"   Первые ошибки: {ada2.losses_[:5]}")
print(f"   Последние ошибки: {ada2.losses_[-5:]}")

ax[1].plot(range(1, len(ada2.losses_) + 1), 
          ada2.losses_, marker='o', linewidth=2, color='green')
ax[1].set_xlabel('Эпохи')
ax[1].set_ylabel('Среднеквадратичная ошибка')
ax[1].set_title('Adaline - скорость обучения 0.0001\n(слишком маленькая - медленная сходимость)')
ax[1].grid(True, alpha=0.3)

# Добавляем аннотацию для медленной сходимости
if ada2.losses_[-1] < ada2.losses_[0]:
    improvement = (ada2.losses_[0] - ada2.losses_[-1]) / ada2.losses_[0] * 100
    ax[1].annotate(f'Ошибка уменьшается\nно очень медленно\nУлучшение: {improvement:.1f}%', 
                   xy=(len(ada2.losses_), ada2.losses_[-1]),
                   xytext=(len(ada2.losses_)*0.6, ada2.losses_[0]*0.8),
                   arrowprops=dict(arrowstyle='->', color='green'),
                   fontsize=10, color='green', ha='center')

plt.tight_layout()
plt.show()

# Дополнительный анализ
print("\nАнализ результатов:")
print("=" * 50)

# Проверяем расходимость
if ada1.losses_[-1] > ada1.losses_[0]:
    print("❌ η=0.1: АЛГОРИТМ РАСХОДИТСЯ")
    print("   • Ошибка растет с каждой эпохой")
    print("   • Веса уходят в бесконечность")
    print("   • Градиентный спуск 'перепрыгивает' минимум")
else:
    print("✅ η=0.1: Алгоритм сходится")

# Проверяем медленную сходимость
improvement_ada2 = (ada2.losses_[0] - ada2.losses_[-1]) / ada2.losses_[0] * 100
if improvement_ada2 < 10:
    print("⚠️  η=0.0001: ОЧЕНЬ МЕДЛЕННАЯ СХОДИМОСТЬ")
    print(f"   • Улучшение за 15 эпох: {improvement_ada2:.1f}%")
    print("   • Потребуются сотни эпох для сходимости")
else:
    print("✅ η=0.0001: Алгоритм сходится нормально")

# Рекомендация
print("\n💡 РЕКОМЕНДАЦИЯ:")
print("   Для этих данных оптимальная скорость обучения: η ≈ 0.001-0.01")
print("   • η=0.1 - слишком большая (расходимость)")
print("   • η=0.0001 - слишком маленькая (медленная сходимость)")
print("   • η=0.01 - хороший компромисс между скоростью и стабильностью")

# Показываем веса для анализа
print(f"\nВеса после обучения:")
print(f"η=0.1:  {ada1.w_} (расходятся)")
print(f"η=0.0001: {ada2.w_} (почти не изменились)")
