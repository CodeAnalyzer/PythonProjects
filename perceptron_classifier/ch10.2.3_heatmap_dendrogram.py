"""
Глава 10.2.3: Прикрепление дендрограмм к тепловой карте

Дендрограммы часто используются в сочетании с тепловой картой,
что позволяет сопоставлять значения в массиве данных с цветовым кодом.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, dendrogram

# Генерируем данные (как в предыдущем примере)
np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
X = np.random.random_sample([5, 3]) * 10
df = pd.DataFrame(X, columns=variables, index=labels)

print("=== ИСХОДНЫЕ ДАННЫЕ ===")
print(df)
print()

# Выполняем иерархическую кластеризацию
row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')

print("=== МАТРИЦА СВЯЗЕЙ ===")
print(pd.DataFrame(row_clusters,
                   columns=['row label 1', 'row label 2', 'distance', 'no. of items in clust.'],
                   index=[f'cluster {i+1}' for i in range(row_clusters.shape[0])]))
print()

# Шаг 1: Создаем фигуру и дендрограмму
fig = plt.figure(figsize=(10, 8), facecolor='white')
axd = fig.add_axes([0.09, 0.1, 0.2, 0.6])
row_dendr = dendrogram(row_clusters, orientation='left')

# Шаг 2: Переупорядочиваем данные в соответствии с метками кластеризации
df_rowclust = df.iloc[row_dendr['leaves'][::-1]]

print("=== ПЕРЕУПОРЯДОЧЕННЫЕ ДАННЫЕ ===")
print(df_rowclust)
print()

# Шаг 3: Создаем тепловую карту
axm = fig.add_axes([0.23, 0.1, 0.6, 0.6])
cax = axm.matshow(df_rowclust,
                  interpolation='nearest',
                  cmap='hot_r')

# Шаг 4: Изменяем внешний вид дендрограммы
axd.set_xticks([])
axd.set_yticks([])
for i in axd.spines.values():
    i.set_visible(False)

# Добавляем цветную линейку
fig.colorbar(cax)

# Назначаем имена признаков и записей данных
axm.set_xticklabels([' '] + list(df_rowclust.columns))
axm.set_yticklabels([' '] + list(df_rowclust.index))

plt.title('Тепловая карта с прикрепленной дендрограммой')
plt.tight_layout()
plt.savefig('heatmap_dendrogram.png', dpi=300, bbox_inches='tight')
print("Сохранена тепловая карта с дендрограммой: heatmap_dendrogram.png")
plt.show()

print()
print("Примечание:")
print("- Дендрограмма слева показывает иерархическую структуру кластеров")
print("- Тепловая карта справа показывает значения признаков с цветовым кодированием")
print("- Строки тепловой карты упорядочены в соответствии с дендрограммой")
print("- Цветовая схема 'hot_r' (hot reversed): темные цвета - низкие значения, светлые - высокие")
