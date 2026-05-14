"""
Глава 10.1.4: Использование метода локтя для нахождения оптимального количества кластеров

Метод локтя (elbow method) помогает оценить оптимальное количество кластеров k.
Идея состоит в том, чтобы определить значение k, при котором искажение (SSE)
начинает уменьшаться наиболее медленно - это и есть "локоть".
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

# Вычисляем искажение (distortion/SSE) для различных значений k
distortions = []
k_values = range(1, 11)

for i in k_values:
    km = KMeans(n_clusters=i,
                init='k-means++',
                n_init=10,
                max_iter=300,
                random_state=0)
    km.fit(X)
    distortions.append(km.inertia_)
    print(f"k={i}: Искажение (SSE) = {km.inertia_:.2f}")

print()

# Строим график метода локтя
plt.plot(k_values, distortions, marker='o')
plt.xlabel('Количество кластеров')
plt.ylabel('Искажение (SSE)')
plt.title('Метод локтя для определения оптимального количества кластеров')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('elbow_method.png', dpi=300, bbox_inches='tight')
print("Сохранен график метода локтя: elbow_method.png")
plt.show()

# Анализ метода локтя
print("=== АНАЛИЗ МЕТОДА ЛОКТЯ ===")
print()

# Вычисляем относительное уменьшение искажения
relative_decrease = []
for i in range(1, len(distortions)):
    decrease = (distortions[i-1] - distortions[i]) / distortions[i-1]
    relative_decrease.append(decrease)
    print(f"k={i} -> k={i+1}: Уменьшение искажения = {decrease:.4f} ({decrease*100:.2f}%)")

print()

# Находим "локоть" - точку, где уменьшение искажения становится наиболее медленным
# Обычно это точка, где относительное уменьшение становится менее 50%
elbow_candidate = None
for i, decrease in enumerate(relative_decrease):
    if decrease < 0.5:
        elbow_candidate = i + 1
        break

if elbow_candidate:
    print(f"Локоть обнаружен при k = {elbow_candidate}")
else:
    # Если явного локтя нет, выбираем точку с максимальным уменьшением второго производной
    # Простой эвристический подход: выбираем k, где уменьшение становится минимальным
    elbow_candidate = np.argmin(relative_decrease) + 1
    print(f"Локоть неявный, выбран k = {elbow_candidate}")

print()

# Визуализация с отмеченным локтем
plt.plot(k_values, distortions, marker='o', label='Искажение')
if elbow_candidate:
    plt.axvline(x=elbow_candidate, color='red', linestyle='--', 
                label=f'Локоть (k={elbow_candidate})')
plt.xlabel('Количество кластеров')
plt.ylabel('Искажение (SSE)')
plt.title('Метод локтя с отмеченным оптимальным k')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('elbow_method_marked.png', dpi=300, bbox_inches='tight')
print("Сохранен график метода локтя с отметкой: elbow_method_marked.png")
plt.show()

print()
print("Примечание: Метод локтя - эвристический подход.")
print("Оптимальное значение k следует подтверждать дополнительными методами,")
print("например, силуэтными графиками или анализом предметной области.")
