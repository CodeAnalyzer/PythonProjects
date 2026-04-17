"""
Раздел 4.2. Работа с категориальными данными
Учебные примеры из книги "Python Machine Learning"
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# 4.2.1. Создание DataFrame с демонстрационными данными
print("=== 4.2.1. Создание DataFrame ===")
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']
])
df.columns = ['color', 'size', 'price', 'classlabel']
print(df)
print()

# 4.2.2. Сопоставление порядковых признаков
print("=== 4.2.2. Сопоставление порядковых признаков (size) ===")
size_mapping = {
    'XL': 3,
    'L': 2,
    'M': 1
}
df['size'] = df['size'].map(size_mapping)
print("DataFrame после сопоставления size:")
print(df)
print()

# Обратное отображение
inv_size_mapping = {v: k for k, v in size_mapping.items()}
print("Обратное отображение size:")
print(df['size'].map(inv_size_mapping))
print()

# 4.2.3. Кодирование меток классов
print("=== 4.2.3. Кодирование меток классов ===")

# Способ 1: ручное сопоставление
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
print("Сопоставление классов:", class_mapping)
df['classlabel'] = df['classlabel'].map(class_mapping)
print("DataFrame после кодирования меток классов:")
print(df)
print()

# Обратное отображение классов
inv_class_mapping = {v: k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print("DataFrame после обратного отображения:")
print(df)
print()

# Способ 2: использование LabelEncoder
print("=== Использование LabelEncoder ===")
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print("Закодированные метки классов:", y)
print("Обратное преобразование:", class_le.inverse_transform(y))
print()

# 4.2.4. Позиционное кодирование номинальных признаков
print("=== 4.2.4. Позиционное кодирование (One-Hot Encoding) ===")

# Неправильный способ - использование LabelEncoder для номинальных признаков
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print("Неправильный способ (LabelEncoder для color):")
print(X)
print("Проблема: предполагается порядок blue=0 < green=1 < red=2")
print()

# Правильный способ - OneHotEncoder
print("Правильный способ - OneHotEncoder:")
X = df[['color', 'size', 'price']].values
color_ohe = OneHotEncoder()
X_color = color_ohe.fit_transform(X[:, 0].reshape(-1, 1)).toarray()
print("One-Hot Encoding для color:")
print(X_color)
print()

# Использование ColumnTransformer для выборочного преобразования
print("=== ColumnTransformer ===")
X = df[['color', 'size', 'price']].values
c_transf = ColumnTransformer([
    ('onehot', OneHotEncoder(), [0]),
    ('nothing', 'passthrough', [1, 2])
])
X_transformed = c_transf.fit_transform(X).astype(float)
print("Результат ColumnTransformer:")
print(X_transformed)
print()

# Использование pandas get_dummies
print("=== pandas get_dummies ===")
print("Полное позиционное кодирование:")
print(pd.get_dummies(df[['price', 'color', 'size']]))
print()

print("С удалением первого столбца (drop_first=True):")
print(pd.get_dummies(df[['price', 'color', 'size']], drop_first=True))
print()

# OneHotEncoder с drop='first'
print("=== OneHotEncoder с drop='first' ===")
color_ohe = OneHotEncoder(categories='auto', drop='first')
c_transf = ColumnTransformer([
    ('onehot', color_ohe, [0]),
    ('nothing', 'passthrough', [1, 2])
])
X_transformed = c_transf.fit_transform(X).astype(float)
print("Результат с drop='first':")
print(X_transformed)
print()

# 4.2.5. Пороговое кодирование порядковых признаков
print("=== 4.2.5. Пороговое кодирование порядковых признаков ===")

# Создаем исходный DataFrame
df_threshold = pd.DataFrame([
    ['green', 'M', 10.1, 'class2'],
    ['red', 'L', 13.5, 'class1'],
    ['blue', 'XL', 15.3, 'class2']
])
df_threshold.columns = ['color', 'size', 'price', 'classlabel']
print("Исходный DataFrame:")
print(df_threshold)
print()

# Пороговое кодирование признака size
df_threshold['x > M'] = df_threshold['size'].apply(lambda x: 1 if x in {'L', 'XL'} else 0)
df_threshold['x > L'] = df_threshold['size'].apply(lambda x: 1 if x == 'XL' else 0)
del df_threshold['size']
print("DataFrame после порогового кодирования:")
print(df_threshold)
