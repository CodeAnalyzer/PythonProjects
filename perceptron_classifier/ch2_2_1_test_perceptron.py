import numpy as np
import matplotlib.pyplot as plt
from ch2_2_1_perceptron import Perceptron

# Создаем простые тестовые данные
# Пример: AND логическая операция
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([0, 0, 0, 1])  # AND: только [1,1] дает 1

print("Тестовые данные:")
print("X =", X)
print("y =", y)

# Создаем и обучаем персептрон
ppn = Perceptron(eta=0.1, n_iter=10, random_state=1)
ppn.fit(X, y)

print("\nРезультаты обучения:")
print(f"Веса: {ppn.w_}")
print(f"Смещение: {ppn.b_}")
print(f"Ошибки по эпохам: {ppn.errors_}")

# Тестируем предсказания
print("\nПредсказания:")
for xi in X:
    prediction = ppn.predict(xi)
    print(f"Вход: {xi} -> Предсказание: {prediction}")

# Проверяем точность
predictions = ppn.predict(X)
accuracy = np.mean(predictions == y) * 100
print(f"\nТочность: {accuracy:.1f}%")

# Визуализируем кривую обучения
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Эпохи')
plt.ylabel('Количество ошибок')
plt.title('Кривая обучения персептрона')
plt.grid(True)
plt.show()

print("\n✅ Тест завершен!")
