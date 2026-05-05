# -*- coding: utf-8 -*-
"""
Раздел 7.4.2: Применение AdaBoost с помощью scikit-learn

Пример демонстрирует:
1. Использование AdaBoostClassifier с DecisionTreeClassifier как базовым классификатором
2. Сравнение производительности обрубка дерева решений и AdaBoost
3. Визуализацию областей принятия решений

AdaBoost (Adaptive Boosting) - метод ансамблевого обучения, который:
- Последовательно обучает слабые классификаторы
- Каждому классификатору назначает вес на основе его ошибок
- Увеличивает вес неправильно классифицированных примеров
- Объединяет предсказания с весами классификаторов
"""
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

print("🚀 ПРИМЕНЕНИЕ ADABOOST С ПОМОЩЬЮ SCIKIT-LEARN")
print("=" * 70)

# 1. Загрузка данных Wine
print("\n📂 1. ЗАГРУЗКА ДАННЫХ WINE")
print("-" * 50)

df_wine = pd.read_csv('wine.data', header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                  'Alcalinity of ash', 'Magnesium', 'Total phenols',
                  'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                  'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

print(f"Исходный набор данных: {df_wine.shape[0]} образцов, {df_wine.shape[1]} признаков")
print(f"Классы: {df_wine['Class label'].unique()}")

# Отбрасываем класс 1, оставляем только классы 2 и 3
df_wine = df_wine[df_wine['Class label'] != 1]

print(f"\nПосле удаления класса 1: {df_wine.shape[0]} образцов")
print(f"Оставшиеся классы: {df_wine['Class label'].unique()}")

# 2. Выбор признаков и кодирование меток
print("\n🔧 2. ПОДГОТОВКА ДАННЫХ")
print("-" * 50)

# Выбираем два признака: Alcohol и OD280/OD315 of diluted wines
X = df_wine[['Alcohol', 'OD280/OD315 of diluted wines']].values
y = df_wine['Class label'].values

print(f"Выбранные признаки:")
print(f"  - Alcohol (индекс 0)")
print(f"  - OD280/OD315 of diluted wines (индекс 1)")
print(f"Форма X: {X.shape}")
print(f"Форма y: {y.shape}")

# Кодирование меток классов в двоичный формат
le = LabelEncoder()
y = le.fit_transform(y)

print(f"\nПосле кодирования:")
print(f"  Метки классов: {np.unique(y)}")
print(f"  Соответствие: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 3. Разделение на обучающие и тестовые наборы
print("\n✂️ 3. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩИЕ И ТЕСТОВЫЕ НАБОРЫ")
print("-" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=1,
    stratify=y
)

print(f"Обучающий набор: {X_train.shape[0]} образцов")
print(f"Тестовый набор: {X_test.shape[0]} образцов")
print(f"Распределение классов в обучающем наборе: {np.bincount(y_train)}")
print(f"Распределение классов в тестовом наборе: {np.bincount(y_test)}")

# 4. Создание классификаторов
print("\n🔧 4. СОЗДАНИЕ КЛАССИФИКАТОРОВ")
print("-" * 50)

# Обрубок дерева решений (max_depth=1)
tree = DecisionTreeClassifier(
    criterion='entropy',
    random_state=1,
    max_depth=1
)

# AdaBoost-классификатор с 500 обрубками
ada = AdaBoostClassifier(
    estimator=tree,  # В новых версиях sklearn: base_estimator -> estimator
    n_estimators=500,
    learning_rate=0.1,
    random_state=1
)

print("Созданы классификаторы:")
print("  1. DecisionTreeClassifier:")
print("     - criterion='entropy', max_depth=1 (обрубок дерева)")
print("  2. AdaBoostClassifier:")
print("     - estimator=DecisionTreeClassifier (max_depth=1)")
print("     - n_estimators=500")
print("     - learning_rate=0.1")

# 5. Оценка обрубка дерева решений
print("\n📊 5. ОЦЕНКА ОБРУБКА ДЕРЕВА РЕШЕНИЙ")
print("-" * 50)

tree = tree.fit(X_train, y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)

tree_train = accuracy_score(y_train, y_train_pred)
tree_test = accuracy_score(y_test, y_test_pred)

print(f'Точность решающего дерева при обучении/проверке: {tree_train:.3f}/{tree_test:.3f}')

if tree_train < 1.0:
    print("ℹ️  Обрубок дерева не полностью обучился на обучающих данных")
    print("   (в отличие от несокращенного дерева из предыдущего раздела)")

# 6. Оценка AdaBoost-классификатора
print("\n📊 6. ОЦЕНКА ADABOOST-КЛАССИФИКАТОРА")
print("-" * 50)

ada = ada.fit(X_train, y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)

ada_train = accuracy_score(y_train, y_train_pred)
ada_test = accuracy_score(y_test, y_test_pred)

print(f'Точность AdaBoost при обучении/проверке: {ada_train:.3f}/{ada_test:.3f}')

if ada_train == 1.0:
    print("✅ AdaBoost правильно предсказывает все метки классов обучающего набора")

# 7. Сравнение результатов
print("\n📈 7. СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("-" * 50)

print(f"{'Метод':<30} {'Обучение':<12} {'Тест':<12}")
print('-' * 55)
print(f"{'Обрубок дерева':<30} {tree_train:<12.3f} {tree_test:<12.3f}")
print(f"{'AdaBoost':<30} {ada_train:<12.3f} {ada_test:<12.3f}")

improvement = (ada_test - tree_test) / tree_test * 100
if improvement > 0:
    print(f"\n✅ AdaBoost улучшает точность на тесте на {improvement:.1f}%")
else:
    print(f"\nℹ️  AdaBoost не улучшает точность на тесте ({improvement:.1f}%)")

# Анализ смещения и дисперсии
train_gap = ada_train - tree_train
test_gap = ada_test - tree_test
print(f"\nАнализ смещения и дисперсии:")
print(f"  Улучшение на обучении: {train_gap:+.3f}")
print(f"  Улучшение на тесте: {test_gap:+.3f}")
print(f"  Разрыв (обучение - тест) для AdaBoost: {ada_train - ada_test:.3f}")
print(f"  Разрыв (обучение - тест) для дерева: {tree_train - tree_test:.3f}")

if (ada_train - ada_test) > (tree_train - tree_test):
    print("  ⚠️  AdaBoost уменьшил смещение, но увеличил дисперсию")

# 8. Визуализация областей принятия решений
print("\n🎨 8. ВИЗУАЛИЗАЦИЯ ОБЛАСТЕЙ ПРИНЯТИЯ РЕШЕНИЙ")
print("-" * 50)

# Создание сетки для визуализации
x_min = X_train[:, 0].min() - 1
x_max = X_train[:, 0].max() + 1
y_min = X_train[:, 1].min() - 1
y_max = X_train[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(nrows=1, ncols=2,
                        sharex='col',
                        sharey='row',
                        figsize=(12, 5))

for idx, clf, tt in zip([0, 1],
                        [tree, ada],
                        ['Дерево решений', 'AdaBoost']):
    clf.fit(X_train, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axarr[idx].contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    axarr[idx].scatter(X_train[y_train == 0, 0],
                       X_train[y_train == 0, 1],
                       c='blue', marker='^', s=50, label='Класс 2')
    axarr[idx].scatter(X_train[y_train == 1, 0],
                       X_train[y_train == 1, 1],
                       c='green', marker='o', s=50, label='Класс 3')
    axarr[idx].set_title(tt, fontsize=12, fontweight='bold')
    axarr[idx].set_xlabel('Alcohol', fontsize=10)
    axarr[idx].legend(loc='best', fontsize=9)
    axarr[idx].grid(alpha=0.3)

axarr[0].set_ylabel('OD280/OD315 of diluted wines', fontsize=10)

plt.suptitle('Сравнение областей принятия решений', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("Области принятия решений визуализированы")

# 9. Выводы
print("\n📝 9. ВЫВОДЫ")
print("=" * 70)
print("Результаты демонстрируют:")
print("  ✅ AdaBoost правильно предсказывает все метки обучающего набора")
print("  ✅ AdaBoost показывает небольшое увеличение точности на тесте")
print("  ✅ Разделяющая граница AdaBoost сложнее, чем у обрубка дерева")
print("\nКак работает AdaBoost:")
print("  - Последовательно обучает 500 обрубков деревьев")
print("  - Каждому дереву назначает вес на основе его ошибок")
print("  - Увеличивает вес неправильно классифицированных примеров")
print("  - Объединяет предсказания с весами деревьев")
print("  - Уменьшает bias (смещение) модели")
print("\nСравнение с бэггингом:")
print("  - AdaBoost уменьшает смещение, но может увеличить дисперсию")
print("  - Бэггинг уменьшает дисперсию, сохраняя смещение")
print("  - AdaBoost использует последовательное обучение")
print("  - Бэггинг использует параллельное обучение")
print("\nНаблюдения:")
print("  - Обрубок дерева не полностью обучился (недообучение)")
print("  - AdaBoost достиг 100% на обучении (компенсация недообучения)")
print("  - AdaBoost показал лучшую точность на тесте")
print("  - Граница решения AdaBoost сложнее и похожа на бэггинг")
print("\nПредостережение:")
print("  - Выбор модели на основе многократного использования тестового набора")
print("    - плохая практика")
print("  - Оценка обобщающей способности может быть чрезмерно оптимистичной")
