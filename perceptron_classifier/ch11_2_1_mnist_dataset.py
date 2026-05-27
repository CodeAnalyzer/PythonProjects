"""
Глава 11.2.1. Получение и подготовка набора данных MNIST

Набор данных MNIST был создан на основе двух наборов данных Национального института
стандартов и технологий США (NIST). Обучающий набор состоит из цифр,
написанных руками 250 разных людей, среди которых 50% - учащиеся старших классов
и 50% - сотрудники Бюро переписи населения.
"""

import numpy as np
import matplotlib.pyplot as plt
import gzip
import os
from sklearn.model_selection import train_test_split

# Функция для загрузки локальных файлов MNIST
def load_mnist_local(path):
    """Загрузка MNIST из локальных файлов .gz"""
    
    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 784)
    
    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data
    
    # Загрузка обучающих данных
    print("Загрузка обучающих данных...")
    X_train = load_images(os.path.join(path, 'train-images-idx3-ubyte.gz'))
    y_train = load_labels(os.path.join(path, 'train-labels-idx1-ubyte.gz'))
    
    # Загрузка тестовых данных
    print("Загрузка тестовых данных...")
    X_test = load_images(os.path.join(path, 't10k-images-idx3-ubyte.gz'))
    y_test = load_labels(os.path.join(path, 't10k-labels-idx1-ubyte.gz'))
    
    # Объединение в один набор (как в оригинальном MNIST)
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([y_train, y_test], axis=0)
    
    return X, y

# Загрузка набора данных MNIST из локальных файлов
mnist_path = r'D:\GITHUB\PythonProjects\MNIST'
print("Загрузка набора данных MNIST из локальных файлов...")
X, y = load_mnist_local(mnist_path)

print(f"Размерность X: {X.shape}")
print(f"Размерность y: {y.shape}")

# Изображения в MNIST имеют размер 28x28 пикселов
# Пиксельная матрица 28x28 развернута в одномерные векторы-строки
# (784 на строку или изображение)

# Масштабирование значений пикселов к диапазону от -1 до 1
# (первоначально было от 0 до 255)
print("\nМасштабирование пикселов к диапазону [-1, 1]...")
X = ((X / 255.) - .5) * 2

# Визуализация примеров цифр от 0 до 9
print("\nВизуализация примеров цифр от 0 до 9...")
fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()

for i in range(10):
    img = X[y == i][0].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.savefig('mnist_digits_0_9.png', dpi=300)
plt.show()

# Визуализация нескольких примеров одной и той же цифры (цифра 7)
print("\nВизуализация 25 примеров цифры 7...")
fig, ax = plt.subplots(nrows=5, ncols=5, sharex=True, sharey=True)
ax = ax.flatten()

for i in range(25):
    img = X[y == 7][i].reshape(28, 28)
    ax[i].imshow(img, cmap='Greys')

ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
plt.savefig('mnist_digit_7_examples.png', dpi=300)
plt.show()

# Разделение набора данных на обучающий, валидационный и тестовый сегменты
print("\nРазделение набора данных на train/valid/test...")
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=10000, random_state=123, stratify=y
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp, test_size=5000, random_state=123, stratify=y_temp
)

print(f"Размер обучающего набора: {X_train.shape}")
print(f"Размер валидационного набора: {X_valid.shape}")
print(f"Размер тестового набора: {X_test.shape}")

# Дополнительная проверка распределения классов
print("\nРаспределение классов в обучающем наборе:")
unique, counts = np.unique(y_train, return_counts=True)
for digit, count in zip(unique, counts):
    print(f"Цифра {digit}: {count} примеров")
