"""
Глава 10.2.4: Агломеративная кластеризация с помощью scikit-learn

AgglomerativeClustering позволяет выбирать количество возвращаемых кластеров,
что полезно для обрезания иерархического дерева кластера.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering

# Генерируем данные (как в предыдущем примере)
np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
X = np.random.random_sample([5, 3]) * 10
df = pd.DataFrame(X, columns=variables, index=labels)

print("=== ИСХОДНЫЕ ДАННЫЕ ===")
print(df)
print()

# Кластеризация с n_clusters=3
print("=== КЛАСТЕРИЗАЦИЯ С n_clusters=3 ===")
ac = AgglomerativeClustering(n_clusters=3,
                             linkage='complete')
labels_3 = ac.fit_predict(X)
print(f"Метки кластеров: {labels_3}")
print()

# Вывод распределения по кластерам
print("Распределение по кластерам:")
for i in range(3):
    cluster_indices = [idx for idx, label in enumerate(labels_3) if label == i]
    cluster_names = [labels[idx] for idx in cluster_indices]
    print(f"  Кластер {i}: {cluster_names}")
print()

# Кластеризация с n_clusters=2
print("=== КЛАСТЕРИЗАЦИЯ С n_clusters=2 ===")
ac2 = AgglomerativeClustering(n_clusters=2,
                              linkage='complete')
labels_2 = ac2.fit_predict(X)
print(f"Метки кластеров: {labels_2}")
print()

# Вывод распределения по кластерам
print("Распределение по кластерам:")
for i in range(2):
    cluster_indices = [idx for idx, label in enumerate(labels_2) if label == i]
    cluster_names = [labels[idx] for idx in cluster_indices]
    print(f"  Кластер {i}: {cluster_names}")
print()

# Сравнение результатов
print("=== СРАВНЕНИЕ РЕЗУЛЬТАТОВ ===")
print(f"{'Наблюдение':<15} {'n_clusters=3':<15} {'n_clusters=2':<15}")
print("-" * 45)
for i, label in enumerate(labels):
    print(f"{label:<15} {labels_3[i]:<15} {labels_2[i]:<15}")
print()

print("Интерпретация:")
print("- При n_clusters=3:")
print("  - ID_0 и ID_4 в одном кластере (метка 1)")
print("  - ID_1 и ID_2 в одном кластере (метка 0)")
print("  - ID_3 в отдельном кластере (метка 2)")
print()
print("- При n_clusters=2:")
print("  - ID_0, ID_3 и ID_4 в одном кластере (метка 0)")
print("  - ID_1 и ID_2 в другом кластере (метка 1)")
print()
print("Примечание:")
print("- Результаты согласуются с дендрограммой из предыдущего примера")
print("- ID_3 больше похож на ID_0 и ID_4, чем на ID_1 и ID_2")
print("- При n_clusters=2 ID_3 правильно назначен к кластеру с ID_0 и ID_4")

# Визуализация кластеризации
print("\n=== ВИЗУАЛИЗАЦИЯ КЛАСТЕРИЗАЦИИ ===")

# Создаем фигуру с двумя подграфиками
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# График для n_clusters=3
colors_3 = ['lightgreen', 'orange', 'lightblue']
for i in range(3):
    cluster_indices = [idx for idx, label in enumerate(labels_3) if label == i]
    cluster_names = [labels[idx] for idx in cluster_indices]
    ax1.scatter(X[cluster_indices, 0], X[cluster_indices, 1], 
                c=colors_3[i], s=100, edgecolor='black', label=f'Кластер {i}')
    for idx in cluster_indices:
        ax1.annotate(labels[idx], (X[idx, 0], X[idx, 1]), 
                    xytext=(5, 5), textcoords='offset points')

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_title('Кластеризация с n_clusters=3')
ax1.legend()
ax1.grid(True, alpha=0.3)

# График для n_clusters=2
colors_2 = ['lightgreen', 'orange']
for i in range(2):
    cluster_indices = [idx for idx, label in enumerate(labels_2) if label == i]
    cluster_names = [labels[idx] for idx in cluster_indices]
    ax2.scatter(X[cluster_indices, 0], X[cluster_indices, 1], 
                c=colors_2[i], s=100, edgecolor='black', label=f'Кластер {i}')
    for idx in cluster_indices:
        ax2.annotate(labels[idx], (X[idx, 0], X[idx, 1]), 
                    xytext=(5, 5), textcoords='offset points')

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('Кластеризация с n_clusters=2')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('agglomerative_clustering_comparison.png', dpi=300, bbox_inches='tight')
print("Сохранен график сравнения кластеризации: agglomerative_clustering_comparison.png")
plt.show()
