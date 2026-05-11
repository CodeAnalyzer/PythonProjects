"""
Глава 9.2.1: Загрузка набора данных Ames Housing в DataFrame

Набор данных Ames Housing состоит из 2930 примеров и 80 признаков.
Мы будем работать с подмножеством признаков:
- Overall Qual: общая оценка материала и отделки дома (1-10)
- Overall Cond: общая оценка состояния дома (1-10)
- Gr Liv Area: жилая площадь над землей (кв. футы)
- Central Air: наличие центрального кондиционера (Y/N)
- Total Bsmt SF: общая площадь подвала (кв. футы)
- SalePrice: цена продажи ($)
"""

import pandas as pd

# Определяем столбцы, которые будем использовать
columns = ['Overall Qual', 'Overall Cond', 'Gr Liv Area',
           'Central Air', 'Total Bsmt SF', 'SalePrice']

# Загружаем данные из локального файла (разделитель - табуляция)
df = pd.read_csv('AmesHousing.txt',
                 sep='\t',
                 usecols=columns)

print("Первые 5 строк набора данных:")
print(df.head())
print()

# Проверяем размеры DataFrame
print(f"Размеры DataFrame: {df.shape}")
print(f"Количество строк: {df.shape[0]}")
print(f"Количество столбцов: {df.shape[1]}")
print()

# Проверяем типы данных
print("Типы данных:")
print(df.dtypes)
print()

# Преобразуем переменную Central Air из строк в целые числа
# 'Y' -> 1, 'N' -> 0
df['Central Air'] = df['Central Air'].map({'Y': 1, 'N': 0})

print("После преобразования Central Air:")
print(df['Central Air'].head())
print()

# Проверяем наличие пропущенных значений
print("Пропущенные значения в каждом столбце:")
print(df.isnull().sum())
print()

# Удаляем записи с пропущенными значениями
df = df.dropna(axis=0)

print("После удаления пропущенных значений:")
print(df.isnull().sum())
print()

print(f"Финальный размер DataFrame: {df.shape}")
print()

# Сохраняем очищенный DataFrame для последующего использования
df.to_csv('ames_housing_clean.csv', index=False)
print("Очищенные данные сохранены в 'ames_housing_clean.csv'")
