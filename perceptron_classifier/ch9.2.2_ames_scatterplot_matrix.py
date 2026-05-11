"""
Глава 9.2.2: Визуализация важных характеристик набора данных

Исследовательский анализ данных (EDA) с использованием scatterplotmatrix из mlxtend.
Матрица диаграмм рассеяния позволяет визуально отобразить попарные корреляции
между различными признаками в одном месте.
"""

import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix

# Загружаем очищенные данные из предыдущего раздела
df = pd.read_csv('ames_housing_clean.csv')

print("Размер набора данных:", df.shape)
print("\nСтолбцы:", df.columns.tolist())
print("\nПервые 5 строк:")
print(df.head())
print()

# Создаем матрицу диаграмм рассеяния
# В mlxtend 0.24.0 API совместим с 0.19.0 для scatterplotmatrix
scatterplotmatrix(df.values, figsize=(12, 10),
                  names=df.columns, alpha=0.5)
plt.tight_layout()
plt.savefig('ames_scatterplot_matrix.png', dpi=300, bbox_inches='tight')
print("Сохранена матрица диаграмм рассеяния: ames_scatterplot_matrix.png")
plt.show()

# Дополнительный анализ: гистограммы распределений каждого признака
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, column in enumerate(df.columns):
    axes[idx].hist(df[column], bins=30, edgecolor='black', alpha=0.7)
    axes[idx].set_xlabel(column, fontsize=10)
    axes[idx].set_ylabel('Частота', fontsize=10)
    axes[idx].set_title(f'Распределение: {column}', fontsize=11, fontweight='bold')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ames_distributions.png', dpi=300, bbox_inches='tight')
print("Сохранены гистограммы распределений: ames_distributions.png")
plt.show()

print("\n=== АНАЛИЗ ВИЗУАЛИЗАЦИИ ===")
print("Матрица диаграмм рассеяния показывает:")
print("  - Попарные зависимости между признаками")
print("  - Распределения каждого признака (на диагонали)")
print("  - Наличие выбросов")
print()
print("Ключевые наблюдения:")
print("  - Gr Liv Area имеет линейную зависимость с SalePrice")
print("  - Overall Qual также коррелирует с SalePrice")
print("  - В SalePrice есть выбросы (экстремально высокие цены)")
