"""
Раздел 4.5.3. Разреженные решения с регуляризацией L1
Учебные примеры из книги "Python Machine Learning"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

# Загрузка набора данных Wine
print("=== Загрузка набора данных Wine ===")
df_wine = pd.read_csv('D:/GITHUB/PythonProjects/perceptron_classifier/wine.data', header=None)

df_wine.columns = [
    'Class label', 'Alcohol', 'Malic acid', 'Ash',
    'Alcalinity of ash', 'Magnesium', 'Total phenols',
    'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'
]

print("Размер набора данных:", df_wine.shape)
print()

# Разделение на обучающие и тестовые наборы
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# Стандартизация данных
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Логистическая регрессия с L1 регуляризацией
print("=== Логистическая регрессия с L1 регуляризацией ===")
lr = LogisticRegression(penalty='l1', C=1.0,
                         solver='liblinear',
                         random_state=0)
lr_ovr = OneVsRestClassifier(lr)
lr_ovr.fit(X_train_std, y_train)

print(f"Точность при обучении: {lr_ovr.score(X_train_std, y_train):.3f}")
print(f"Точность при тестировании: {lr_ovr.score(X_test_std, y_test):.3f}")
print()

print("Пересечения (intercept_):")
print(lr_ovr.estimators_[0].intercept_)
print()

print("Веса (coef_):")
for i, estimator in enumerate(lr_ovr.estimators_):
    print(f"Класс {i+1}: {estimator.coef_}")
print()

# Анализ разреженности весов
print("=== Анализ разреженности весов ===")
for i, estimator in enumerate(lr_ovr.estimators_):
    class_coef = estimator.coef_[0]
    non_zero = np.count_nonzero(class_coef)
    total = len(class_coef)
    print(f"Класс {i+1}: {non_zero} ненулевых весов из {total} ({non_zero/total*100:.1f}%)")
print()

# Построение пути регуляризации
print("=== Построение пути регуляризации ===")
fig = plt.figure(figsize=(10, 6))
ax = plt.subplot(111)

colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black',
          'pink', 'lightgreen', 'lightblue', 'gray', 'indigo', 'orange']

weights, params = [], []
for c in np.arange(-4., 6.):
    lr = LogisticRegression(penalty='l1', C=10.**c,
                           solver='liblinear',
                           random_state=0)
    lr_ovr = OneVsRestClassifier(lr)
    lr_ovr.fit(X_train_std, y_train)
    weights.append(lr_ovr.estimators_[1].coef_[0])  # Веса для класса 2
    params.append(10.**c)

weights = np.array(weights)

for column, color in zip(range(weights.shape[1]), colors):
    plt.plot(params, weights[:, column],
             label=df_wine.columns[column + 1],
             color=color)

plt.axhline(0, color='black', linestyle='--', linewidth=3)
plt.xlim([10**(-5), 10**5])
plt.ylabel('Весовой коэффициент')
plt.xlabel('C (обратный уровень регуляризации)')
plt.xscale('log')
plt.legend(loc='upper left', bbox_to_anchor=(1.38, 1.03),
           ncol=1, fancybox=True)
plt.title('Путь регуляризации L1 для класса 2')
plt.tight_layout()
plt.savefig('regularization_l1_path.png', dpi=150, bbox_inches='tight')
print("График пути регуляризации сохранен как 'regularization_l1_path.png'")
plt.show()

# Демонстрация влияния C на разреженность
print("\n=== Влияние параметра C на разреженность ===")
print("C - обратный уровень регуляризации (меньше C = сильнее регуляризация)")
print()

c_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
print("C\t\tКласс 1\tКласс 2\tКласс 3")
for c in c_values:
    lr = LogisticRegression(penalty='l1', C=c,
                           solver='liblinear',
                           random_state=0)
    lr_ovr = OneVsRestClassifier(lr)
    lr_ovr.fit(X_train_std, y_train)
    
    non_zero_counts = []
    for estimator in lr_ovr.estimators_:
        class_coef = estimator.coef_[0]
        non_zero = np.count_nonzero(class_coef)
        non_zero_counts.append(non_zero)
    
    print(f"{c:.3f}\t\t{non_zero_counts[0]}\t\t{non_zero_counts[1]}\t\t{non_zero_counts[2]}")
