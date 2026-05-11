"""
Глава 9.2.3: Просмотр отношений с помощью матрицы корреляции

Матрица корреляции количественно оценивает линейные взаимосвязи между переменными.
Коэффициент корреляции Пирсона (r) находится в диапазоне от -1 до 1:
- r = 1: полная положительная корреляция
- r = 0: отсутствие корреляции
- r = -1: полная отрицательная корреляция
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import heatmap

# Загружаем очищенные данные
df = pd.read_csv('ames_housing_clean.csv')

print("Размер набора данных:", df.shape)
print("\nСтолбцы:", df.columns.tolist())
print()

# Вычисляем матрицу корреляции с помощью NumPy
# Транспонируем значения (.T), чтобы получить корреляцию между столбцами
cm = np.corrcoef(df.values.T)

print("Матрица корреляции:")
print(cm)
print()

# Создаем тепловую карту матрицы корреляции с помощью mlxtend
hm = heatmap(cm, row_names=df.columns, column_names=df.columns)
plt.tight_layout()
plt.savefig('ames_correlation_heatmap.png', dpi=300, bbox_inches='tight')
print("Сохранена тепловая карта: ames_correlation_heatmap.png")
plt.show()

# Дополнительный анализ: вывод корреляций с SalePrice
print("\n=== КОРРЕЛЯЦИИ С SALEPRICE ===")
correlations = pd.Series(cm[df.columns.get_loc('SalePrice')], index=df.columns)
correlations_sorted = correlations.abs().sort_values(ascending=False)

print("Признаки, отсортированные по абсолютной корреляции с SalePrice:")
for feature, corr in correlations_sorted.items():
    actual_corr = correlations[feature]
    direction = "положительная" if actual_corr > 0 else "отрицательная"
    print(f"  {feature}: {actual_corr:.3f} ({direction})")

print()

# Выделение признаков с высокой корреляцией
high_corr = correlations_sorted[correlations_sorted >= 0.5].drop('SalePrice')
print(f"Признаки с высокой корреляцией (|r| ≥ 0.5) с SalePrice:")
for feature in high_corr.index:
    print(f"  - {feature}: {correlations[feature]:.3f}")

print()

# Выбор признака для простой линейной регрессии
best_feature = correlations_sorted.index[1]  # Пропускаем сам SalePrice
print(f"Наибольшая корреляция с SalePrice: {best_feature} ({correlations[best_feature]:.3f})")
print(f"Этот признак будет хорошим кандидатом для простой линейной регрессии.")
