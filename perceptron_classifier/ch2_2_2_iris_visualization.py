import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Загрузка данных
df = pd.read_csv('iris.data', header=None, encoding='utf-8')
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Выбираем setosa и versicolor (первые 100 образцов)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', 0, 1)

# Извлекаем длину чашелистика (столбец 0) и длину лепестка (столбец 2)
X = df.iloc[0:100, [0, 2]].values

print("Форма матрицы X:", X.shape)
print("Форма вектора y:", y.shape)
print("\nПервые 5 строк X:")
print(X[:5])
print("\nПервые 5 меток y:")
print(y[:5])

# Отображаем данные
plt.figure(figsize=(10, 6))

# Setosa (первые 50 образцов)
plt.scatter(X[:50, 0], X[:50, 1], 
            color='red', marker='o', label='Setosa')

# Versicolor (следующие 50 образцов)
plt.scatter(X[50:100, 0], X[50:100, 1], 
            color='blue', marker='s', label='Versicolor')

plt.xlabel('Длина чашелистика [см]')
plt.ylabel('Длина лепестка [см]')
plt.title('Набор данных Iris: Setosa vs Versicolor')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

print("\n✅ Визуализация завершена!")
print("Красные круги: Setosa (класс 0)")
print("Синие квадраты: Versicolor (класс 1)")
