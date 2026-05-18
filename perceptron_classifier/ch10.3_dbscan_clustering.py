"""
Глава 10.3: Обнаружение областей высокой плотности с помощью DBSCAN

DBSCAN не предполагает сферическую форму кластеров, как метод k-средних.
Кроме того, DBSCAN не обязательно назначает каждую точку кластеру и может удалять шумовые точки.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# Создаем набор данных серповидной формы
X, y = make_moons(n_samples=200,
                  noise=0.05,
                  random_state=0)

print("Размер набора данных:", X.shape)
print()

# Визуализируем исходные данные
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.title('Исходные данные (серповидная форма)')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('dbscan_original_data.png', dpi=300, bbox_inches='tight')
print("Сохранен график исходных данных: dbscan_original_data.png")
plt.show()

# Сравнение k-means и агломеративной кластеризации
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# K-means кластеризация
km = KMeans(n_clusters=2, random_state=0)
y_km = km.fit_predict(X)

ax1.scatter(X[y_km == 0, 0],
            X[y_km == 0, 1],
            c='lightblue',
            edgecolor='black',
            marker='o',
            s=40,
            label='Кластер 1')
ax1.scatter(X[y_km == 1, 0],
            X[y_km == 1, 1],
            c='red',
            edgecolor='black',
            marker='s',
            s=40,
            label='Кластер 2')
ax1.set_title('Кластеризация по k-средним')
ax1.set_xlabel('Признак 1')
ax1.set_ylabel('Признак 2')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Агломеративная кластеризация
ac = AgglomerativeClustering(n_clusters=2,
                            linkage='complete')
y_ac = ac.fit_predict(X)

ax2.scatter(X[y_ac == 0, 0],
            X[y_ac == 0, 1],
            c='lightblue',
            edgecolor='black',
            marker='o',
            s=40,
            label='Кластер 1')
ax2.scatter(X[y_ac == 1, 0],
            X[y_ac == 1, 1],
            c='red',
            edgecolor='black',
            marker='s',
            s=40,
            label='Кластер 2')
ax2.set_title('Агломеративная кластеризация')
ax2.set_xlabel('Признак 1')
ax2.set_ylabel('Признак 2')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dbscan_comparison_traditional.png', dpi=300, bbox_inches='tight')
print("Сохранен график сравнения традиционных методов: dbscan_comparison_traditional.png")
plt.show()

# DBSCAN кластеризация
db = DBSCAN(eps=0.2,
            min_samples=5,
            metric='euclidean')
y_db = db.fit_predict(X)

print("=== РЕЗУЛЬТАТЫ DBSCAN ===")
print(f"Количество кластеров: {len(set(y_db)) - (1 if -1 in y_db else 0)}")
print(f"Количество шумовых точек: {list(y_db).count(-1)}")
print(f"Уникальные метки: {set(y_db)}")
print()

plt.scatter(X[y_db == 0, 0],
            X[y_db == 0, 1],
            c='lightblue',
            edgecolor='black',
            marker='o',
            s=40,
            label='Кластер 1')
plt.scatter(X[y_db == 1, 0],
            X[y_db == 1, 1],
            c='red',
            edgecolor='black',
            marker='s',
            s=40,
            label='Кластер 2')
if -1 in y_db:
    plt.scatter(X[y_db == -1, 0],
                X[y_db == -1, 1],
                c='green',
                edgecolor='black',
                marker='x',
                s=40,
                label='Шум')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.title('Кластеризация DBSCAN')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('dbscan_clustering.png', dpi=300, bbox_inches='tight')
print("Сохранен график DBSCAN кластеризации: dbscan_clustering.png")
plt.show()

print("Примечание:")
print("- K-means не смог разделить два кластера серповидной формы")
print("- Агломеративная кластеризация также не справилась с этими сложными формами")
print("- DBSCAN успешно распознал группы точек в форме полумесяцев")
print("- DBSCAN может кластеризовать данные произвольной формы")
print("- DBSCAN может идентифицировать шумовые точки")
