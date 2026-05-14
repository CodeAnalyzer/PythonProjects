"""
Глава 10.1.1: Кластеризация методом k-средних с использованием scikit-learn
Глава 10.1.2: Сравнение init='random' и init='k-means++'

Алгоритм k-средних относится к категории кластеризации на основе прототипов.
Каждый кластер представлен прототипом - центроидом (средним) похожих точек.

k-means++ - более разумный способ размещения начальных центроидов.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# Создаем синтетический набор данных с помощью make_blobs
X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

print("Размер набора данных:", X.shape)
print()

# Визуализируем исходные данные
plt.scatter(X[:, 0],
            X[:, 1],
            c='white',
            marker='o',
            edgecolor='black',
            s=50)
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.title('Исходные данные (150 точек, 3 кластера)')
plt.grid()
plt.tight_layout()
plt.savefig('kmeans_original_data.png', dpi=300, bbox_inches='tight')
print("Сохранен график исходных данных: kmeans_original_data.png")
plt.show()

# Применяем алгоритм k-средних
km = KMeans(n_clusters=3,
            init='random',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)

y_km = km.fit_predict(X)

print("=== РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ ===")
print(f"Количество кластеров: {km.n_clusters}")
print(f"Количество итераций: {km.n_iter_}")
print(f"Сходимость достигнута: {km.n_iter_ < km.max_iter}")
print()

# Визуализируем кластеры с центроидами
plt.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            s=50, c='lightgreen',
            marker='s', edgecolor='black',
            label='Кластер 1')
plt.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            s=50, c='orange',
            marker='o', edgecolor='black',
            label='Кластер 2')
plt.scatter(X[y_km == 2, 0],
            X[y_km == 2, 1],
            s=50, c='lightblue',
            marker='v', edgecolor='black',
            label='Кластер 3')
plt.scatter(km.cluster_centers_[:, 0],
            km.cluster_centers_[:, 1],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='Центроиды')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend(scatterpoints=1)
plt.grid()
plt.title('Результаты кластеризации k-средних')
plt.tight_layout()
plt.savefig('kmeans_clusters.png', dpi=300, bbox_inches='tight')
print("Сохранен график кластеров: kmeans_clusters.png")
plt.show()

# Выводим координаты центроидов
print("=== КООРДИНАТЫ ЦЕНТРОИДОВ ===")
for i, center in enumerate(km.cluster_centers_):
    print(f"Кластер {i + 1}: ({center[0]:.2f}, {center[1]:.2f})")
print()

# Выводим внутрикластерную сумму квадратов (SSE)
print(f"Внутрикластерная сумма квадратов (SSE): {km.inertia_:.2f}")
print()

# Распределение точек по кластерам
print("=== РАСПРЕДЕЛЕНИЕ ТОЧЕК ПО КЛАСТЕРАМ ===")
unique, counts = np.unique(y_km, return_counts=True)
for cluster, count in zip(unique, counts):
    print(f"Кластер {cluster}: {count} точек")

print("\n" + "="*60)
print("=== СРАВНЕНИЕ С K-MEANS++ ===")
print("="*60 + "\n")

# Применяем алгоритм k-средних с инициализацией k-means++
km_pp = KMeans(n_clusters=3,
               init='k-means++',
               n_init=10,
               max_iter=300,
               tol=1e-04,
               random_state=0)

y_km_pp = km_pp.fit_predict(X)

print("=== РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ (K-MEANS++) ===")
print(f"Количество кластеров: {km_pp.n_clusters}")
print(f"Количество итераций: {km_pp.n_iter_}")
print(f"Сходимость достигнута: {km_pp.n_iter_ < km_pp.max_iter}")
print()

# Визуализируем кластеры с центроидами (k-means++)
plt.scatter(X[y_km_pp == 0, 0],
            X[y_km_pp == 0, 1],
            s=50, c='lightgreen',
            marker='s', edgecolor='black',
            label='Кластер 1')
plt.scatter(X[y_km_pp == 1, 0],
            X[y_km_pp == 1, 1],
            s=50, c='orange',
            marker='o', edgecolor='black',
            label='Кластер 2')
plt.scatter(X[y_km_pp == 2, 0],
            X[y_km_pp == 2, 1],
            s=50, c='lightblue',
            marker='v', edgecolor='black',
            label='Кластер 3')
plt.scatter(km_pp.cluster_centers_[:, 0],
            km_pp.cluster_centers_[:, 1],
            s=250, marker='*',
            c='red', edgecolor='black',
            label='Центроиды')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend(scatterpoints=1)
plt.grid()
plt.title('Результаты кластеризации k-средних++')
plt.tight_layout()
plt.savefig('kmeans_plusplus_clusters.png', dpi=300, bbox_inches='tight')
print("Сохранен график кластеров (k-means++): kmeans_plusplus_clusters.png")
plt.show()

# Выводим координаты центроидов (k-means++)
print("=== КООРДИНАТЫ ЦЕНТРОИДОВ (K-MEANS++) ===")
for i, center in enumerate(km_pp.cluster_centers_):
    print(f"Кластер {i + 1}: ({center[0]:.2f}, {center[1]:.2f})")
print()

# Выводим внутрикластерную сумму квадратов (k-means++)
print(f"Внутрикластерная сумма квадратов (SSE): {km_pp.inertia_:.2f}")
print()

# Распределение точек по кластерам (k-means++)
print("=== РАСПРЕДЕЛЕНИЕ ТОЧЕК ПО КЛАСТЕРАМ (K-MEANS++) ===")
unique_pp, counts_pp = np.unique(y_km_pp, return_counts=True)
for cluster, count in zip(unique_pp, counts_pp):
    print(f"Кластер {cluster}: {count} точек")
print()

# Сравнение результатов
print("=== СРАВНЕНИЕ РЕЗУЛЬТАТОВ ===")
print(f"{'Метрика':<30} {'init=random':<20} {'init=k-means++':<20}")
print("-"*70)
print(f"{'Итерации':<30} {km.n_iter_:<20} {km_pp.n_iter_:<20}")
print(f"{'SSE (Inertia)':<30} {km.inertia_:<20.2f} {km_pp.inertia_:<20.2f}")
print(f"{'Разница SSE':<30} {'-':<20} {km.inertia_ - km_pp.inertia_:<20.2f}")
print()

if km_pp.inertia_ < km.inertia_:
    print("k-means++ показал лучшее качество (меньшее SSE)")
else:
    print("init=random показал лучшее качество (меньшее SSE)")
print()

print("Примечание: k-means++ является методом по умолчанию в scikit-learn,")
print("так как он обычно обеспечивает лучшую сходимость и качество кластеризации.")
