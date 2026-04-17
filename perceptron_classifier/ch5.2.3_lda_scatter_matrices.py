"""
Разделы 5.2.3-5.2.6. LDA: вычисление матриц разброса, выбор дискриминантов и реализация в scikit-learn
Учебный пример из книги "Python Machine Learning"
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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
print("Раздел 5.2.3. Вычисление матриц разброса для LDA")
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

# Стандартизация данных (уже выполнена в разд. 5.1.2)
print("\n=== 2. Стандартизация данных ===")
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

print("Средние значения признаков после стандартизации (train):")
print(np.round(X_train_std.mean(axis=0), 6))
print(f"\nФорма X_train_std: {X_train_std.shape}")

# Вычисление средних векторов для каждого класса
print("\n=== 3. Вычисление средних векторов ===")
np.set_printoptions(precision=4)

mean_vecs = []
for label in range(1, 4):
    mean_vecs.append(np.mean(X_train_std[y_train == label], axis=0))
    print(f'MV {label}: {mean_vecs[label - 1]}\n')

# Вычисление матрицы внутриклассового разброса (немасштабированной)
print("\n=== 4. Матрица внутриклассового разброса (немасштабированная) ===")
d = 13  # количество признаков
S_W = np.zeros((d, d))

for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.zeros((d, d))
    for row in X_train_std[y_train == label]:
        row, mv = row.reshape(d, 1), mv.reshape(d, 1)
        class_scatter += (row - mv).dot((row - mv).T)
    S_W += class_scatter

print(f'Матрица внутриклассового разброса: {S_W.shape[0]}x{S_W.shape[1]}')

# Проверка распределения меток классов
print("\n=== 5. Распределение меток класса ===")
print('Распределение меток класса:', np.bincount(y_train)[1:])
print("Примечание: распределение неравномерное, необходимо масштабирование")

# Вычисление масштабированной матрицы внутриклассового разброса (ковариационной матрицы)
print("\n=== 6. Масштабированная матрица внутриклассового разброса ===")
d = 13  # количество признаков
S_W = np.zeros((d, d))

for label, mv in zip(range(1, 4), mean_vecs):
    class_scatter = np.cov(X_train_std[y_train == label].T)
    S_W += class_scatter

print(f'Масшт. матрица внутриклассового разброса: {S_W.shape[0]}x{S_W.shape[1]}')

# Вычисление матрицы межклассового разброса
print("\n=== 7. Матрица межклассового разброса ===")
mean_overall = np.mean(X_train_std, axis=0)
mean_overall = mean_overall.reshape(d, 1)

d = 13  # количество признаков
S_B = np.zeros((d, d))

for i, mean_vec in enumerate(mean_vecs):
    n = X_train_std[y_train == i + 1, :].shape[0]
    mean_vec = mean_vec.reshape(d, 1)  # вектор столбца
    S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

print(f'Матрица межклассового разброса: {S_B.shape[0]}x{S_B.shape[1]}')

# Вывод результатов
print("\n=== 8. Результаты ===")
print("Средние векторы по классам:")
for i, mv in enumerate(mean_vecs, start=1):
    print(f"Класс {i}: {mv}")

print(f"\nРазмерность матрицы внутриклассового разброса S_W: {S_W.shape}")
print(f"Размерность матрицы межклассового разброса S_B: {S_B.shape}")

print("\nПервые 5x5 элементов матрицы S_W:")
print(np.round(S_W[:5, :5], 4))

print("\nПервые 5x5 элементов матрицы S_B:")
print(np.round(S_B[:5, :5], 4))


# ============================================================================
# Раздел 5.2.4. Выбор линейных дискриминантов для нового подпространства признаков
# ============================================================================
print("\n" + "=" * 72)
print("Раздел 5.2.4. Выбор линейных дискриминантов")
print("=" * 72)

# Решение обобщенной задачи нахождения собственных значений матрицы S_W^(-1) * S_B
print("\n=== 9. Вычисление собственных значений и собственных векторов ===")
eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))

print("Собственные значения (все):")
print(eigen_vals)

# Сортировка собственных значений в порядке убывания
print("\n=== 10. Сортировка собственных значений по убыванию ===")
eigen_pairs = [
    (np.abs(eigen_vals[i]), eigen_vecs[:, i])
    for i in range(len(eigen_vals))
]
eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

print('Собственные значения по убыванию:\n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])

print("\nПримечание: количество линейных дискриминантов не превышает c-1,")
print("где c - количество меток классов (в данном случае 3-1=2)")
print("Собственные значения с 3 по 13 близки к нулю из-за особенностей")
print("арифметики с плавающей запятой в NumPy")

# Вычисление дискриминируемости (аналог объясненной дисперсии в PCA)
print("\n=== 11. Дискриминируемость линейных дискриминантов ===")
tot = sum(eigen_vals.real)
discr = [(i / tot) for i in sorted(eigen_vals.real, reverse=True)]
cum_discr = np.cumsum(discr)

print("Индивидуальная дискриминируемость:")
for idx, d in enumerate(discr, start=1):
    print(f"LD{idx}: {d:.8f}")

print("\nНакопительная дискриминируемость:")
for idx, cd in enumerate(cum_discr, start=1):
    print(f"LD{idx}: {cd:.8f}")

# Визуализация дискриминируемости
print("\n=== 12. Визуализация дискриминируемости ===")
plt.figure(figsize=(10, 6))
plt.bar(range(1, 14), discr, align='center', label='Индивидуальная дискриминируемость')
plt.step(range(1, 14), cum_discr, where='mid', label='Накопительная дискриминируемость')
plt.ylabel('Коэффициент дискриминируемости')
plt.xlabel('Линейные дискриминанты')
plt.ylim((-0.1, 1.1))
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('d:\\GITHUB\\PythonProjects\\perceptron_classifier\\lda_discriminability.png', dpi=150)
print("График дискриминируемости сохранен как: lda_discriminability.png")
plt.show()

print("\nПервые два линейных дискриминанта извлекают 100% полезной информации")
print("о разделении классов в обучающем наборе данных Wine.")

# Создание матрицы преобразования W из двух наиболее дискриминируемых собственных векторов
print("\n=== 13. Создание матрицы преобразования W ===")
w = np.hstack((
    eigen_pairs[0][1][:, np.newaxis].real,
    eigen_pairs[1][1][:, np.newaxis].real
))
print('Матрица W:\n', w)
print(f'\nФорма матрицы W: {w.shape}')


# ============================================================================
# Раздел 5.2.5. Проецирование точек данных на новое функциональное пространство
# ============================================================================
print("\n" + "=" * 72)
print("Раздел 5.2.5. Проецирование точек данных на новое функциональное пространство")
print("=" * 72)

# Преобразование обучающих данных с помощью матрицы W
print("\n=== 14. Преобразование обучающих данных ===")
X_train_lda = X_train_std.dot(w)
print(f"Форма X_train_lda: {X_train_lda.shape}")
print("Первые 5 преобразованных образцов:")
print(np.round(X_train_lda[:5], 4))

# Визуализация результатов в пространстве LD1-LD2
print("\n=== 15. Визуализация обучающего набора в пространстве LD1-LD2 ===")
colors = ['r', 'b', 'g']
markers = ['o', 's', '^']

plt.figure(figsize=(8, 6))
for label, c, m in zip(np.unique(y_train), colors, markers):
    plt.scatter(
        X_train_lda[y_train == label, 0],
        X_train_lda[y_train == label, 1] * (-1),
        c=c,
        label=f'Class {label}',
        marker=m,
        edgecolor='black',
        alpha=0.8
    )

plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('d:\\GITHUB\\PythonProjects\\perceptron_classifier\\lda_wine_projection.png', dpi=150)
print("График LDA-проекции сохранен как: lda_wine_projection.png")
plt.show()

print("\nТри класса набора данных Wine теперь идеально линейно разделимы")
print("в новом подпространстве признаков.")


# ============================================================================
# Раздел 5.2.6. Реализация LDA при помощи scikit-learn
# ============================================================================
print("\n" + "=" * 72)
print("Раздел 5.2.6. Реализация LDA при помощи scikit-learn")
print("=" * 72)

# Создание и обучение LDA
print("\n=== 16. LDA через scikit-learn ===")
lda = LDA(n_components=2)
X_train_lda_sklearn = lda.fit_transform(X_train_std, y_train)

print(f"Форма X_train_lda_sklearn: {X_train_lda_sklearn.shape}")
print("Первые 5 преобразованных образцов (sklearn LDA):")
print(np.round(X_train_lda_sklearn[:5], 4))

# Обучение логистической регрессии на преобразованных данных
print("\n=== 17. Логистическая регрессия на LDA-преобразованных данных ===")
lr = LogisticRegression(random_state=1, solver='lbfgs')
lr = lr.fit(X_train_lda_sklearn, y_train)

print(f"Точность LogisticRegression на train (LDA): {lr.score(X_train_lda_sklearn, y_train):.4f}")

# Визуализация областей решений для обучающего набора
plt.figure(figsize=(8, 6))
plot_decision_regions(X_train_lda_sklearn, y_train, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('d:\\GITHUB\\PythonProjects\\perceptron_classifier\\lda_sklearn_train_decision_regions.png', dpi=150)
print("График областей решений train (LDA + sklearn) сохранен как: lda_sklearn_train_decision_regions.png")
plt.show()

# Преобразование тестового набора и оценка
print("\n=== 18. Оценка на тестовом наборе ===")
X_test_lda_sklearn = lda.transform(X_test_std)
print(f"Форма X_test_lda_sklearn: {X_test_lda_sklearn.shape}")
print(f"Точность LogisticRegression на test (LDA): {lr.score(X_test_lda_sklearn, y_test):.4f}")

# Визуализация областей решений для тестового набора
plt.figure(figsize=(8, 6))
plot_decision_regions(X_test_lda_sklearn, y_test, classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('d:\\GITHUB\\PythonProjects\\perceptron_classifier\\lda_sklearn_test_decision_regions.png', dpi=150)
print("График областей решений test (LDA + sklearn) сохранен как: lda_sklearn_test_decision_regions.png")
plt.show()

print("\nКлассификатор логистической регрессии может с идеальной точностью")
print("классифицировать экземпляры в тестовом наборе данных, используя только")
print("двумерное подпространство признаков вместо исходных 13 признаков Wine.")
