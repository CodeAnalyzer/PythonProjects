"""
Раздел 4.3. Разделение набора данных на обучающие и тестовые наборы
Учебные примеры из книги "Python Machine Learning"
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Загрузка набора данных Wine из локального файла
print("=== Загрузка набора данных Wine ===")
df_wine = pd.read_csv('D:/GITHUB/PythonProjects/perceptron_classifier/wine.data', header=None)

df_wine.columns = [
    'Class label', 'Alcohol', 'Malic acid', 'Ash',
    'Alcalinity of ash', 'Magnesium', 'Total phenols',
    'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'
]

print("Class labels:", np.unique(df_wine['Class label']))
print("\nПервые 5 строк набора данных:")
print(df_wine.head())
print("\nРазмер набора данных:", df_wine.shape)
print()

# Разделение на признаки и метки классов
print("=== Разделение на обучающие и тестовые наборы ===")
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

print("Размер массива признаков X:", X.shape)
print("Размер массива меток y:", y.shape)
print()

# Разделение на обучающие и тестовые наборы с помощью train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=0,
    stratify=y
)

print("Размер обучающего набора X_train:", X_train.shape)
print("Размер тестового набора X_test:", X_test.shape)
print("Размер обучающих меток y_train:", y_train.shape)
print("Размер тестовых меток y_test:", y_test.shape)
print()

# Проверка стратификации (пропорции классов)
print("=== Проверка стратификации ===")
print("Пропорции классов в исходном наборе:")
print(np.bincount(y) / len(y))
print("\nПропорции классов в обучающем наборе:")
print(np.bincount(y_train) / len(y_train))
print("\nПропорции классов в тестовом наборе:")
print(np.bincount(y_test) / len(y_test))
