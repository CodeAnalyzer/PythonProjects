"""
Глава 10.2.2: Выполнение иерархической кластеризации с матрицей расстояний

Агломеративная иерархическая кластеризация с использованием матрицы расстояний
и метода полной связи (complete linkage).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
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

# Вычисляем матрицу расстояний
print("=== МАТРИЦА РАССТОЯНИЙ ===")
row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')),
                         columns=labels, index=labels)
print(row_dist)
print()

# Неправильный подход: использование квадратной матрицы расстояний
print("=== НЕПРАВИЛЬНЫЙ ПОДХОД (использование квадратной матрицы) ===")
print("ВНИМАНИЕ: Это приведет к неправильным результатам!")
row_clusters_wrong = linkage(row_dist, method='complete', metric='euclidean')
pd.DataFrame(row_clusters_wrong,
             columns=['row label 1', 'row label 2', 'distance', 'no. of items in clust.'],
             index=[f'cluster {i+1}' for i in range(row_clusters_wrong.shape[0])])
print()

# Правильный подход 1: использование сжатой матрицы расстояний
print("=== ПРАВИЛЬНЫЙ ПОДХОД 1 (сжатая матрица расстояний) ===")
row_clusters = linkage(pdist(df, metric='euclidean'), method='complete')
print("Матрица связей:")
print(pd.DataFrame(row_clusters,
                   columns=['row label 1', 'row label 2', 'distance', 'no. of items in clust.'],
                   index=[f'cluster {i+1}' for i in range(row_clusters.shape[0])]))
print()

# Правильный подход 2: использование полной входной матрицы примеров
print("=== ПРАВИЛЬНЫЙ ПОДХОД 2 (полная входная матрица) ===")
row_clusters2 = linkage(df.values, method='complete', metric='euclidean')
print("Матрица связей:")
print(pd.DataFrame(row_clusters2,
                   columns=['row label 1', 'row label 2', 'distance', 'no. of items in clust.'],
                   index=[f'cluster {i+1}' for i in range(row_clusters2.shape[0])]))
print()

# Сравнение результатов
print("=== СРАВНЕНИЕ РЕЗУЛЬТАТОВ ===")
print("Правильный подход 1 (сжатая матрица):")
print(row_clusters)
print()
print("Правильный подход 2 (полная матрица):")
print(row_clusters2)
print()
if np.allclose(row_clusters, row_clusters2):
    print("Результаты идентичны ✓")
else:
    print("Результаты различаются")

# Построение дендрограммы
plt.figure(figsize=(10, 6))
row_dendr = dendrogram(row_clusters,
                       labels=labels,
                       color_threshold=np.inf)  # черная дендрограмма
plt.ylabel('Евклидово расстояние')
plt.title('Дендрограмма агломеративной иерархической кластеризации')
plt.tight_layout()
plt.savefig('hierarchical_dendrogram_black.png', dpi=300, bbox_inches='tight')
print("Сохранена черная дендрограмма: hierarchical_dendrogram_black.png")
plt.show()

# Построение цветной дендрограммы (по умолчанию)
plt.figure(figsize=(10, 6))
row_dendr = dendrogram(row_clusters,
                       labels=labels)
plt.ylabel('Евклидово расстояние')
plt.title('Дендрограмма агломеративной иерархической кластеризации (цветная)')
plt.tight_layout()
plt.savefig('hierarchical_dendrogram_color.png', dpi=300, bbox_inches='tight')
print("Сохранена цветная дендрограмма: hierarchical_dendrogram_color.png")
plt.show()

print()
print("Примечание:")
print("- Первый и второй столбцы матрицы связей обозначают наиболее непохожие элементы в каждом кластере")
print("- Третий столбец показывает расстояние между этими элементами")
print("- Последний столбец содержит количество элементов в каждом кластере")
print("- Дендрограмма показывает, что ID_0 и ID_4 наиболее похожи, за ними следуют ID_1 и ID_2")
