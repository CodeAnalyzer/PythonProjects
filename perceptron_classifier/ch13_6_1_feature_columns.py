import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# ── Загрузка данных Auto MPG ──
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower',
                'Weight', 'Acceleration', 'Model Year', 'Origin']

df = pd.read_csv(url, names=column_names,
                 na_values="?", comment='\t',
                 sep=" ", skipinitialspace=True)

print(f'Исходный размер данных: {df.shape}')

# ── Удаление строк с пропущенными значениями ──
df = df.dropna()
df = df.reset_index(drop=True)
print(f'После удаления NA: {df.shape}')

# ── Разделение на обучающий и тестовый наборы ──
df_train, df_test = train_test_split(df, train_size=0.8, random_state=1)
print(f'Обучающий набор: {df_train.shape}')
print(f'Тестовый набор: {df_test.shape}')

# ── Статистика обучающего набора ──
train_stats = df_train.describe().transpose()
print('\nСтатистика числовых признаков (обучающий набор):')
print(train_stats[['mean', 'std']].head(10))

# ── Стандартизация числовых признаков ──
numeric_column_names = [
    'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration'
]

df_train_norm = df_train.copy()
df_test_norm = df_test.copy()

for col_name in numeric_column_names:
    mean = train_stats.loc[col_name, 'mean']
    std = train_stats.loc[col_name, 'std']
    df_train_norm[col_name] = df_train_norm[col_name].astype(float)
    df_test_norm[col_name] = df_test_norm[col_name].astype(float)
    df_train_norm.loc[:, col_name] = (df_train_norm.loc[:, col_name] - mean) / std
    df_test_norm.loc[:, col_name] = (df_test_norm.loc[:, col_name] - mean) / std

print('\nНормализованные данные (последние 5 строк обучающего набора):')
print(df_train_norm[numeric_column_names].tail())

# ── Группировка Model Year в бакеты (bucketization) ──
boundaries = torch.tensor([73, 76, 79])

v = torch.tensor(df_train_norm['Model Year'].values)
df_train_norm['Model Year Bucketed'] = torch.bucketize(v, boundaries, right=True)

v = torch.tensor(df_test_norm['Model Year'].values)
df_test_norm['Model Year Bucketed'] = torch.bucketize(v, boundaries, right=True)

numeric_column_names.append('Model Year Bucketed')

print('\nРаспределение по бакетам Model Year (обучающий набор):')
print(df_train_norm['Model Year Bucketed'].value_counts().sort_index())

# ── One-hot encoding для категориального признака Origin ──
total_origin = len(set(df_train_norm['Origin']))
print(f'\nКоличество уникальных значений Origin: {total_origin}')

origin_encoded_train = F.one_hot(torch.from_numpy(
    df_train_norm['Origin'].values.copy()) % total_origin, num_classes=total_origin)

x_train_numeric = torch.tensor(df_train_norm[numeric_column_names].values)
x_train = torch.cat([x_train_numeric, origin_encoded_train], 1).float()

origin_encoded_test = F.one_hot(torch.from_numpy(
    df_test_norm['Origin'].values.copy()) % total_origin, num_classes=total_origin)

x_test_numeric = torch.tensor(df_test_norm[numeric_column_names].values)
x_test = torch.cat([x_test_numeric, origin_encoded_test], 1).float()

print(f'\nФорма обучающих признаков: {x_train.shape}')
print(f'Форма тестовых признаков: {x_test.shape}')

# ── Подготовка целевых значений (MPG) ──
y_train = torch.tensor(df_train_norm['MPG'].values).float()
y_test = torch.tensor(df_test_norm['MPG'].values).float()

print(f'\nФорма целевых значений (обучение): {y_train.shape}')
print(f'Форма целевых значений (тест): {y_test.shape}')

# ── Итоговая информация ──
print('\n=== ИТОГОВАЯ СТРУКТУРА ДАННЫХ ===')
print(f'Признаки: {len(numeric_column_names)} числовых + {total_origin} категориальных (one-hot)')
print(f'Всего признаков: {x_train.shape[1]}')
print(f'Образцы: обучение={len(x_train)}, тест={len(x_test)}')
print(f'Цель: MPG (miles per gallon)')
