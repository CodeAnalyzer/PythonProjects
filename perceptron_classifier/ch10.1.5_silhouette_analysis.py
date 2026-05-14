"""
Глава 10.1.5: Количественная оценка качества кластеризации с помощью силуэтных графиков

Силуэтный анализ - графический инструмент для измерения того, насколько плотно
сгруппированы точки данных в кластерах. Силуэтный коэффициент находится в диапазоне от -1 до 1.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

# Создаем синтетический набор данных с помощью make_blobs
X, y = make_blobs(n_samples=150,
                  n_features=2,
                  centers=3,
                  cluster_std=0.5,
                  shuffle=True,
                  random_state=0)

print("Размер набора данных:", X.shape)
print()

# Обучаем k-средних с k=3
km = KMeans(n_clusters=3,
            init='k-means++',
            n_init=10,
            max_iter=300,
            tol=1e-04,
            random_state=0)

y_km = km.fit_predict(X)

print("=== РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ ===")
print(f"Количество кластеров: {km.n_clusters}")
print(f"Количество итераций: {km.n_iter_}")
print(f"Внутрикластерная сумма квадратов (SSE): {km.inertia_:.2f}")
print()

# Вычисляем силуэтные коэффициенты
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]

silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')

print("=== СИЛУЭТНЫЙ АНАЛИЗ ===")
print(f"Средний силуэтный коэффициент: {np.mean(silhouette_vals):.4f}")
print()

# Создаем график силуэтных коэффициентов
y_ax_lower, y_ax_upper = 0, 0
yticks = []

for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters)
    plt.barh(range(y_ax_lower, y_ax_upper),
             c_silhouette_vals,
             height=1.0,
             edgecolor='none',
             color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)

silhouette_avg = np.mean(silhouette_vals)

plt.axvline(silhouette_avg, color="red", linestyle="--",
            label=f'Средний коэффициент = {silhouette_avg:.3f}')
plt.yticks(yticks, cluster_labels + 1)
plt.ylabel('Кластер')
plt.xlabel('Силуэтный коэффициент')
plt.title('Силуэтный график для k-средних (k=3)')
plt.legend()
plt.tight_layout()
plt.savefig('silhouette_analysis_k3.png', dpi=300, bbox_inches='tight')
print("Сохранен силуэтный график: silhouette_analysis_k3.png")
plt.show()

# Анализ силуэтных коэффициентов для каждого кластера
print("=== АНАЛИЗ ПО КЛАСТЕРАМ ===")
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]
    print(f"Кластер {c + 1}:")
    print(f"  Количество точек: {len(c_silhouette_vals)}")
    print(f"  Средний силуэтный коэффициент: {np.mean(c_silhouette_vals):.4f}")
    print(f"  Минимальный силуэтный коэффициент: {np.min(c_silhouette_vals):.4f}")
    print(f"  Максимальный силуэтный коэффициент: {np.max(c_silhouette_vals):.4f}")
    print()

# Интерпретация результатов
print("=== ИНТЕРПРЕТАЦИЯ ===")
print("Силуэтный коэффициент:")
print("  Близко к 1: точка хорошо соответствует своему кластеру")
print("  Близко к 0: точка на границе между кластерами")
print("  Отрицательное значение: точка может быть в неправильном кластере")
print()
print(f"Средний силуэтный коэффициент = {silhouette_avg:.4f}")
if silhouette_avg > 0.7:
    print("Отличная структура кластеров")
elif silhouette_avg > 0.5:
    print("Хорошая структура кластеров")
elif silhouette_avg > 0.25:
    print("Слабая структура кластеров")
else:
    print("Плохая структура кластеров")

print("\n" + "="*60)
print("=== СРАВНЕНИЕ С ПЛОХОЙ КЛАСТЕРИЗАЦИЕЙ (k=2) ===")
print("="*60 + "\n")

# Обучаем k-средних с k=2 (плохая кластеризация для этого набора данных)
km_bad = KMeans(n_clusters=2,
               init='k-means++',
               n_init=10,
               max_iter=300,
               tol=1e-04,
               random_state=0)

y_km_bad = km_bad.fit_predict(X)

print("=== РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ (k=2) ===")
print(f"Количество кластеров: {km_bad.n_clusters}")
print(f"Количество итераций: {km_bad.n_iter_}")
print(f"Внутрикластерная сумма квадратов (SSE): {km_bad.inertia_:.2f}")
print()

# Визуализируем кластеры с центроидами (k=2)
plt.scatter(X[y_km_bad == 0, 0],
            X[y_km_bad == 0, 1],
            s=50, c='lightgreen',
            edgecolor='black',
            marker='s',
            label='Кластер 1')
plt.scatter(X[y_km_bad == 1, 0],
            X[y_km_bad == 1, 1],
            s=50,
            c='orange',
            edgecolor='black',
            marker='o',
            label='Кластер 2')
plt.scatter(km_bad.cluster_centers_[:, 0],
            km_bad.cluster_centers_[:, 1],
            s=250,
            marker='*',
            c='red',
            label='Центроиды')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend()
plt.grid()
plt.title('Кластеризация k-средних (k=2) - неоптимальная')
plt.tight_layout()
plt.savefig('kmeans_k2_clusters.png', dpi=300, bbox_inches='tight')
print("Сохранен график кластеров (k=2): kmeans_k2_clusters.png")
plt.show()

# Вычисляем силуэтные коэффициенты для k=2
cluster_labels_bad = np.unique(y_km_bad)
n_clusters_bad = cluster_labels_bad.shape[0]

silhouette_vals_bad = silhouette_samples(X, y_km_bad, metric='euclidean')

print("=== СИЛУЭТНЫЙ АНАЛИЗ (k=2) ===")
print(f"Средний силуэтный коэффициент: {np.mean(silhouette_vals_bad):.4f}")
print()

# Создаем график силуэтных коэффициентов (k=2)
y_ax_lower, y_ax_upper = 0, 0
yticks = []

for i, c in enumerate(cluster_labels_bad):
    c_silhouette_vals = silhouette_vals_bad[y_km_bad == c]
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)
    color = cm.jet(float(i) / n_clusters_bad)
    plt.barh(range(y_ax_lower, y_ax_upper),
             c_silhouette_vals,
             height=1.0,
             edgecolor='none',
             color=color)
    yticks.append((y_ax_lower + y_ax_upper) / 2.)
    y_ax_lower += len(c_silhouette_vals)

silhouette_avg_bad = np.mean(silhouette_vals_bad)

plt.axvline(silhouette_avg_bad, color="red", linestyle="--",
            label=f'Средний коэффициент = {silhouette_avg_bad:.3f}')
plt.yticks(yticks, cluster_labels_bad + 1)
plt.ylabel('Кластер')
plt.xlabel('Силуэтный коэффициент')
plt.title('Силуэтный график для k-средних (k=2) - плохая кластеризация')
plt.legend()
plt.tight_layout()
plt.savefig('silhouette_analysis_k2.png', dpi=300, bbox_inches='tight')
print("Сохранен силуэтный график (k=2): silhouette_analysis_k2.png")
plt.show()

# Анализ силуэтных коэффициентов для каждого кластера (k=2)
print("=== АНАЛИЗ ПО КЛАСТЕРАМ (k=2) ===")
for i, c in enumerate(cluster_labels_bad):
    c_silhouette_vals = silhouette_vals_bad[y_km_bad == c]
    print(f"Кластер {c + 1}:")
    print(f"  Количество точек: {len(c_silhouette_vals)}")
    print(f"  Средний силуэтный коэффициент: {np.mean(c_silhouette_vals):.4f}")
    print(f"  Минимальный силуэтный коэффициент: {np.min(c_silhouette_vals):.4f}")
    print(f"  Максимальный силуэтный коэффициент: {np.max(c_silhouette_vals):.4f}")
    print()

# Сравнение k=3 и k=2
print("=== СРАВНЕНИЕ k=3 И k=2 ===")
print(f"{'Метрика':<30} {'k=3':<20} {'k=2':<20}")
print("-"*70)
print(f"{'SSE (Inertia)':<30} {km.inertia_:<20.2f} {km_bad.inertia_:<20.2f}")
print(f"{'Средний силуэтный коэффициент':<30} {silhouette_avg:<20.4f} {silhouette_avg_bad:<20.4f}")
print()

if silhouette_avg > silhouette_avg_bad:
    print("k=3 показывает лучшее качество кластеризации (более высокий силуэтный коэффициент)")
else:
    print("k=2 показывает лучшее качество кластеризации (более высокий силуэтный коэффициент)")
print()

print("Примечание: Силуэтные коэффициенты для k=2 имеют заметно разную длину и ширину,")
print("что свидетельствует об относительно плохой или неоптимальной кластеризации.")
