import os
import pandas as pd

# Путь к локальному файлу
file_path = 'iris.data'

print('From local file:', file_path)

# Загрузка данных в DataFrame
df = pd.read_csv(file_path,
                 header=None,
                 encoding='utf-8')

print("\nПоследние 5 строк набора данных Iris:")
print(df.tail())

# Добавим названия столбцов для удобства
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

print("\nНабор данных с названиями столбцов:")
print(df.tail())

print("\nИнформация о наборе данных:")
print(df.info())

print("\nСтатистика по числовым признакам:")
print(df.describe())

print("\nУникальные классы:")
print(df['class'].unique())

print("\nКоличество образцов по классам:")
print(df['class'].value_counts())
