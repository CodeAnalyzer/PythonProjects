"""
Разделы 5.1.2-5.1.5. Пошаговый процесс извлечения основных компонент (PCA)
Учебный пример из книги "Python Machine Learning"
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def plot_decision_regions(X, y, classifier, resolution=0.02):
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=f'Класс {cl}',
            edgecolor='black'
        )


print("=" * 72)
print("Разделы 5.1.2-5.1.5. PCA: ручная реализация и scikit-learn")
print("=" * 72)


# Загрузка набора данных Wine
print("\n=== 1. Загрузка набора данных Wine ===")
df_wine = pd.read_csv('D:/GITHUB/PythonProjects/perceptron_classifier/wine.data', header=None)

df_wine.columns = [
    'Class label', 'Alcohol', 'Malic acid', 'Ash',
    'Alcalinity of ash', 'Magnesium', 'Total phenols',
    'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'
]

print(f"Размер набора данных: {df_wine.shape}")
print("Классы:", np.unique(df_wine['Class label']))

# Разделение на признаки и метки классов
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=0
)

print(f"Размер обучающего набора: {X_train.shape}")
print(f"Размер тестового набора: {X_test.shape}")

# Шаг 1. Стандартизация данных
print("\n=== 2. Шаг 1: стандартизация данных ===")
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

print("Средние значения признаков после стандартизации (train):")
print(np.round(X_train_std.mean(axis=0), 6))
print("\nСтандартные отклонения после стандартизации (train):")
print(np.round(X_train_std.std(axis=0), 6))
print(f"\nФорма X_train_std: {X_train_std.shape}")
print(f"Форма X_test_std: {X_test_std.shape}")

# Шаг 2. Построение ковариационной матрицы
print("\n=== 3. Шаг 2: ковариационная матрица ===")
cov_mat = np.cov(X_train_std.T)
print(f"Форма ковариационной матрицы: {cov_mat.shape}")
print("Первые 5x5 элементов ковариационной матрицы:")
print(np.round(cov_mat[:5, :5], 4))

# Шаг 3. Получение собственных значений и собственных векторов
print("\n=== 4. Шаг 3: собственные значения и собственные векторы ===")
eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)

print("Собственные значения:")
print(np.round(eigen_vals, 8))
print(f"\nФорма матрицы собственных векторов: {eigen_vecs.shape}")
print("Первый собственный вектор:")
print(np.round(eigen_vecs[:, 0], 4))

# Шаг 4. Сортировка собственных значений по убыванию
print("\n=== 5. Шаг 4: сортировка собственных значений ===")
eigen_pairs = [
    (np.abs(eigen_vals[i]), eigen_vecs[:, i])
    for i in range(len(eigen_vals))
]
eigen_pairs.sort(key=lambda k: k[0], reverse=True)

print("Собственные значения в порядке убывания:")
for idx, (eigen_val, _) in enumerate(eigen_pairs, start=1):
    print(f"PC{idx:02d}: {eigen_val:.8f}")

# Доля объясненной дисперсии
print("\n=== 6. Раздел 5.1.3: общая и объясненная дисперсия ===")
tot = np.sum(eigen_vals)
var_exp = [(i / tot) for i in sorted(eigen_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)

for idx, (var, cum_var) in enumerate(zip(var_exp, cum_var_exp), start=1):
    print(
        f"PC{idx:02d}: explained variance = {var:.4f}, "
        f"cumulative explained variance = {cum_var:.4f}"
    )

plt.figure(figsize=(10, 6))
plt.bar(
    range(1, len(var_exp) + 1),
    var_exp,
    align='center',
    label='Отдельные объясненные дисперсии'
)
plt.step(
    range(1, len(cum_var_exp) + 1),
    cum_var_exp,
    where='mid',
    label='Совокупная объясненная дисперсия'
)
plt.ylabel('Доля объясненной дисперсии')
plt.xlabel('Индекс главной компоненты')
plt.xticks(range(1, len(var_exp) + 1))
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('d:\\GITHUB\\PythonProjects\\perceptron_classifier\\pca_wine_explained_variance.png', dpi=150)
print("\nГрафик explained variance сохранен как: pca_wine_explained_variance.png")
plt.show()

# Формирование матрицы проекции из двух главных компонент
print("\n=== 7. Матрица проекции W из двух главных компонент ===")
w = np.hstack((
    eigen_pairs[0][1][:, np.newaxis],
    eigen_pairs[1][1][:, np.newaxis]
))

print("Матрица W:")
print(np.round(w, 4))
print(f"Форма W: {w.shape}")

print("\nПроекция первого стандартизированного объекта x' = xW:")
print(np.round(X_train_std[0].dot(w), 8))

print("\n=== 8. Раздел 5.1.4: преобразование признаков ===")
X_train_pca = X_train_std.dot(w)
print(f"Форма X_train_pca: {X_train_pca.shape}")
print("Первые 5 преобразованных образцов:")
print(np.round(X_train_pca[:5], 4))

print("\n=== 9. Визуализация обучающего набора в пространстве PC1-PC2 ===")
colors = ['r', 'b', 'g']
markers = ['o', 's', '^']

plt.figure(figsize=(8, 6))
for label, color, marker in zip(np.unique(y_train), colors, markers):
    plt.scatter(
        X_train_pca[y_train == label, 0],
        X_train_pca[y_train == label, 1],
        c=color,
        label=f'Class {label}',
        marker=marker,
        edgecolor='black',
        alpha=0.8
    )

plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('d:\\GITHUB\\PythonProjects\\perceptron_classifier\\pca_wine_projection.png', dpi=150)
print("График PCA-проекции сохранен как: pca_wine_projection.png")
plt.show()

print("\n=== 10. Раздел 5.1.5: PCA в scikit-learn ===")
pca = PCA(n_components=2)
lr = LogisticRegression(random_state=1, solver='lbfgs')

X_train_pca_sklearn = pca.fit_transform(X_train_std)
X_test_pca_sklearn = pca.transform(X_test_std)

lr.fit(X_train_pca_sklearn, y_train)

print(f"Форма X_train_pca_sklearn: {X_train_pca_sklearn.shape}")
print(f"Форма X_test_pca_sklearn: {X_test_pca_sklearn.shape}")
print(f"Точность LogisticRegression на train: {lr.score(X_train_pca_sklearn, y_train):.4f}")
print(f"Точность LogisticRegression на test: {lr.score(X_test_pca_sklearn, y_test):.4f}")

plt.figure(figsize=(8, 6))
plot_decision_regions(X_train_pca_sklearn, y_train, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('d:\\GITHUB\\PythonProjects\\perceptron_classifier\\pca_sklearn_train_decision_regions.png', dpi=150)
print("График областей решений train сохранен как: pca_sklearn_train_decision_regions.png")
plt.show()

plt.figure(figsize=(8, 6))
plot_decision_regions(X_test_pca_sklearn, y_test, classifier=lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('d:\\GITHUB\\PythonProjects\\perceptron_classifier\\pca_sklearn_test_decision_regions.png', dpi=150)
print("График областей решений test сохранен как: pca_sklearn_test_decision_regions.png")
plt.show()

print("\n=== 11. Раздел 5.1.6: оценка вклада признаков ===")
feature_names = df_wine.columns[1:]

loadings = eigen_vecs * np.sqrt(eigen_vals)
print("Нагрузки первой главной компоненты (ручной PCA):")
print(np.round(loadings[:, 0], 4))

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(13), loadings[:, 0], align='center')
ax.set_ylabel('Loadings for PC 1')
ax.set_xticks(range(13))
ax.set_xticklabels(feature_names, rotation=90)
plt.ylim([-1, 1])
plt.tight_layout()
plt.savefig('d:\\GITHUB\\PythonProjects\\perceptron_classifier\\pca_manual_loadings_pc1.png', dpi=150)
print("График нагрузок ручного PCA сохранен как: pca_manual_loadings_pc1.png")
plt.show()

sklearn_loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
print("\nНагрузки первой главной компоненты (scikit-learn PCA):")
print(np.round(sklearn_loadings[:, 0], 4))

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(range(13), sklearn_loadings[:, 0], align='center')
ax.set_ylabel('Loadings for PC 1')
ax.set_xticks(range(13))
ax.set_xticklabels(feature_names, rotation=90)
plt.ylim([-1, 1])
plt.tight_layout()
plt.savefig('d:\\GITHUB\\PythonProjects\\perceptron_classifier\\pca_sklearn_loadings_pc1.png', dpi=150)
print("График нагрузок sklearn PCA сохранен как: pca_sklearn_loadings_pc1.png")
plt.show()
