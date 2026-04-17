import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ch2_2_1_perceptron import Perceptron

# Загрузка и подготовка данных
df = pd.read_csv('iris.data', header=None, encoding='utf-8')
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Выбираем setosa и versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)

# Извлекаем признаки: длина чашелистика и длина лепестка
X = df.iloc[0:100, [0, 2]].values

print("Данные для обучения:")
print(f"Форма X: {X.shape}")
print(f"Форма y: {y.shape}")
print(f"Классы: {np.unique(y, return_counts=True)}")

# Создаем и обучаем персептрон
ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)

print(f"\nРезультаты обучения:")
print(f"Веса: {ppn.w_}")
print(f"Смещение: {ppn.b_}")
print(f"Ошибки по эпохам: {ppn.errors_}")

# Строим график ошибок
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(ppn.errors_) + 1), 
         ppn.errors_, 
         marker='o', 
         markersize=8, 
         linewidth=2)

plt.xlabel('Эпохи')
plt.ylabel('Количество ошибок')
plt.title('Кривая обучения персептрона на данных Iris')
plt.grid(True, alpha=0.3)
plt.show()

# Проверяем точность на обучающих данных
predictions = ppn.predict(X)
accuracy = np.mean(predictions == y) * 100
print(f"\nТочность на обучающих данных: {accuracy:.1f}%")

# Показываем предсказания для первых 10 образцов
print("\nПредсказания для первых 10 образцов:")
for i in range(10):
    print(f"Образец {i+1}: X={X[i]}, Истинный={y[i]}, Предсказанный={predictions[i]}")

if ppn.errors_[-1] == 0:
    print("\n✅ Персептрон успешно сошелся!")
else:
    print(f"\n⚠️  Персептрон не сошелся, осталось {ppn.errors_[-1]} ошибок")
