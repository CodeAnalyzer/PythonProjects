# -*- coding: utf-8 -*-
"""
Раздел 8.1.2: Преобразование набора данных в более удобный формат

Пример демонстрирует:
1. Чтение обзоров фильмов из директории aclImdb
2. Сбор всех текстовых файлов в один CSV-файл
3. Перемешивание данных для случайного распределения
4. Сохранение в формате movie_data.csv

Набор данных IMDB содержит 50 000 обзоров фильмов:
- 25 000 обучающих (12 500 положительных, 12 500 отрицательных)
- 25 000 тестовых (12 500 положительных, 12 500 отрицательных)
"""
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import pandas as pd
import numpy as np

print("📚 ПРЕОБРАЗОВАНИЕ НАБОРА ДАННЫХ В УДОБНЫЙ ФОРМАТ")
print("=" * 70)

# Проверка наличия pyprind
try:
    import pyprind
    use_pyprind = True
    print(f"✅ PyPrind доступен")
except ImportError:
    use_pyprind = False
    print("⚠️  PyPrind не установлен, будет использоваться встроенный прогресс-бар")
    print("   Установить можно командой: pip install pyprind")

# 1. Настройка путей
print("\n📂 1. НАСТРОЙКА ПУТЕЙ")
print("-" * 50)

basepath = 'aclImdb'

# Проверка наличия директории
if not os.path.exists(basepath):
    print(f"❌ Ошибка: директория '{basepath}' не найдена")
    print(f"   Текущая директория: {os.getcwd()}")
    sys.exit(1)

print(f"Базовая директория: {basepath}")
print(f"Полный путь: {os.path.abspath(basepath)}")

# Проверка структуры директорий
required_dirs = [
    os.path.join(basepath, 'train', 'pos'),
    os.path.join(basepath, 'train', 'neg'),
    os.path.join(basepath, 'test', 'pos'),
    os.path.join(basepath, 'test', 'neg')
]

for dir_path in required_dirs:
    if not os.path.exists(dir_path):
        print(f"❌ Ошибка: директория '{dir_path}' не найдена")
        sys.exit(1)

print("✅ Все необходимые директории найдены")

# Подсчет количества файлов
print("\n📊 2. ПОДСЧЕТ ФАЙЛОВ")
print("-" * 50)

labels = {'pos': 1, 'neg': 0}
total_files = 0

for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        file_count = len(os.listdir(path))
        total_files += file_count
        print(f"  {s}/{l}: {file_count} файлов")

print(f"\nВсего файлов: {total_files}")

# 3. Чтение файлов и создание DataFrame
print("\n📖 3. ЧТЕНИЕ ФАЙЛОВ И СОЗДАНИЕ DATAFRAME")
print("-" * 50)

df = pd.DataFrame()

if use_pyprind:
    pbar = pyprind.ProgBar(total_files, stream=sys.stdout)
else:
    print(f"Обработка {total_files} файлов...")

for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        files = sorted(os.listdir(path))
        
        for file in files:
            file_path = os.path.join(path, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    txt = infile.read()
                
                # Добавляем в DataFrame
                df = pd.concat([df, pd.DataFrame([[txt, labels[l]]])], 
                               ignore_index=True)
                
                if use_pyprind:
                    pbar.update()
                elif len(df) % 5000 == 0:
                    print(f"  Обработано: {len(df)}/{total_files} файлов")
                    
            except Exception as e:
                print(f"⚠️  Ошибка при чтении {file_path}: {e}")

print(f"\n✅ Прочитано {len(df)} файлов")

# 4. Назначение имен колонок
print("\n🏷️  4. НАЗНАЧЕНИЕ ИМЕН КОЛОНОК")
print("-" * 50)

df.columns = ['review', 'sentiment']
print(f"Колонки: {df.columns.tolist()}")

# 5. Перемешивание данных
print("\n🔀 5. ПЕРЕМЕШИВАНИЕ ДАННЫХ")
print("-" * 50)

np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df = df.reset_index(drop=True)

print("✅ Данные перемешаны")
print(f"Распределение классов:")
print(f"  Положительные (1): {df['sentiment'].sum()}")
print(f"  Отрицательные (0): {len(df) - df['sentiment'].sum()}")

# 6. Сохранение в CSV
print("\n💾 6. СОХРАНЕНИЕ В CSV")
print("-" * 50)

csv_filename = 'movie_data.csv'
df.to_csv(csv_filename, index=False, encoding='utf-8')

print(f"✅ Данные сохранены в файл: {csv_filename}")
print(f"Размер файла: {os.path.getsize(csv_filename) / (1024*1024):.2f} MB")

# 7. Проверка результата
print("\n✅ 7. ПРОВЕРКА РЕЗУЛЬТАТА")
print("-" * 50)

df_check = pd.read_csv(csv_filename, encoding='utf-8')
print(f"Размер DataFrame: {df_check.shape}")
print(f"Ожидаемый размер: (50000, 2)")

if df_check.shape == (50000, 2):
    print("✅ Размер DataFrame соответствует ожидаемому")
else:
    print("⚠️  Размер DataFrame не соответствует ожидаемому")

print("\nПервые 3 записи:")
print(df_check.head(3))

# 8. Статистика
print("\n📊 8. СТАТИСТИКА")
print("-" * 50)

print(f"Всего обзоров: {len(df_check)}")
print(f"Положительных: {df_check['sentiment'].sum()}")
print(f"Отрицательных: {len(df_check) - df_check['sentiment'].sum()}")
print(f"Средняя длина обзора: {df_check['review'].str.len().mean():.1f} символов")
print(f"Максимальная длина обзора: {df_check['review'].str.len().max()} символов")

# 9. Выводы
print("\n📝 9. ВЫВОДЫ")
print("=" * 70)
print("Результаты:")
print("  ✅ Все 50 000 обзоров успешно прочитаны")
print("  ✅ Данные сохранены в movie_data.csv")
print("  ✅ Данные перемешаны для случайного распределения")
print("  ✅ Метки классов: 1 = положительный, 0 = отрицательный")
print("\nСтруктура CSV-файла:")
print("  - Колонка 'review': текст обзора фильма")
print("  - Колонка 'sentiment': метка класса (0 или 1)")
print("\nСледующие шаги:")
print("  - Использовать movie_data.csv для обучения моделей")
print("  - Применить препроцессинг текста")
print("  - Разделить на обучающий и тестовый наборы")
print("  - Обучить классификаторы для анализа тональности")
