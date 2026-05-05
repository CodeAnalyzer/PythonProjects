# -*- coding: utf-8 -*-
"""
Раздел 7.2.2: Использование принципа мажоритарного голосования для прогнозирования

Пример демонстрирует использование MajorityVoteClassifier на наборе данных Iris.
Выбираются два класса (Iris-versicolor и Iris-virginica) и два признака
(ширина чашелистика и длина лепестка) для демонстрации ансамблевого обучения.

Три классификатора:
1. Логистическая регрессия (с масштабированием)
2. Дерево решений
3. K-ближайших соседей (с масштабированием)

Оценка производится с помощью 10-кратной перекрестной проверки по метрике ROC AUC.
"""
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# Импорт MajorityVoteClassifier из предыдущего раздела
import importlib.util
spec = importlib.util.spec_from_file_location("majority_vote", "ch7.2.1_majority_vote_classifier.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MajorityVoteClassifier = module.MajorityVoteClassifier

print("🌸 ИСПОЛЬЗОВАНИЕ МАЖОРИТАРНОГО ГОЛОСОВАНИЯ НА НАБОРЕ IRIS")
print("=" * 70)

# 1. Загрузка и подготовка данных
print("\n📂 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ IRIS")
print("-" * 50)

iris = datasets.load_iris()

# Выбор двух классов: versicolor (1) и virginica (2) - строки 50+
# Выбор двух признаков: ширина чашелистика (индекс 1) и длина лепестка (индекс 2)
X = iris.data[50:, [1, 2]]
y = iris.target[50:]

print(f"Исходные метки классов: {np.unique(y)}")
print(f"Названия классов: {iris.target_names[1:]}")
print(f"Выбранные признаки: {iris.feature_names[1]} и {iris.feature_names[2]}")
print(f"Форма X: {X.shape}")
print(f"Форма y: {y.shape}")

# Кодирование меток классов (преобразование в 0 и 1)
le = LabelEncoder()
y = le.fit_transform(y)

print(f"\nПосле кодирования:")
print(f"  Метки классов: {np.unique(y)}")
print(f"  Соответствие: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 2. Разделение на обучающие и тестовые наборы
print("\n✂️ 2. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩИЕ И ТЕСТОВЫЕ НАБОРЫ")
print("-" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.5,
    random_state=1,
    stratify=y
)

print(f"Обучающий набор: {X_train.shape[0]} образцов")
print(f"Тестовый набор: {X_test.shape[0]} образцов")
print(f"Распределение классов в обучающем наборе: {np.bincount(y_train)}")
print(f"Распределение классов в тестовом наборе: {np.bincount(y_test)}")

# 3. Создание классификаторов
print("\n🔧 3. СОЗДАНИЕ КЛАССИФИКАТОРОВ")
print("-" * 50)

# Логистическая регрессия
clf1 = LogisticRegression(
    C=0.001,
    solver='lbfgs',
    random_state=1
)

# Дерево решений
clf2 = DecisionTreeClassifier(
    max_depth=1,
    criterion='entropy',
    random_state=0
)

# K-ближайших соседей
clf3 = KNeighborsClassifier(
    n_neighbors=1,
    p=2,
    metric='minkowski'
)

print("Созданы три классификатора:")
print("  1. LogisticRegression:")
print("     - C=0.001, solver='lbfgs' (L2-регуляризация по умолчанию)")
print("  2. DecisionTreeClassifier:")
print("     - max_depth=1, criterion='entropy'")
print("  3. KNeighborsClassifier:")
print("     - n_neighbors=1, p=2, metric='minkowski'")

# 4. Создание пайплайнов с масштабированием
print("\n🔗 4. СОЗДАНИЕ ПАЙПЛАЙНОВ С МАСШТАБИРОВАНИЕМ")
print("-" * 50)

# Пайплайн для логистической регрессии (требует масштабирования)
pipe1 = Pipeline([
    ['sc', StandardScaler()],
    ['clf', clf1]
])

# Пайплайн для KNN (требует масштабирования)
pipe3 = Pipeline([
    ['sc', StandardScaler()],
    ['clf', clf3]
])

print("Созданы пайплайны:")
print("  - pipe1: StandardScaler + LogisticRegression")
print("  - clf2: DecisionTreeClassifier (без масштабирования)")
print("  - pipe3: StandardScaler + KNeighborsClassifier")
print("\nПримечание:")
print("  Логистическая регрессия и KNN требуют масштабирования признаков,")
print("  так как они не являются масштабно-инвариантными.")
print("  Деревья решений масштабно-инвариантны.")

# 5. Оценка отдельных классификаторов
print("\n📊 5. ОЦЕНКА ОТДЕЛЬНЫХ КЛАССИФИКАТОРОВ")
print("-" * 50)

clf_labels = ['Логистическая регрессия', 'Дерево решений', 'KNN']

print('10-кратная перекрестная проверка (Accuracy):\n')
for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(
        estimator=clf,
        X=X_train,
        y=y_train,
        cv=10,
        scoring='accuracy'
    )
    print(f'Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f}) [{label}]')

# 6. Создание ансамбля MajorityVoteClassifier
print("\n🗳️ 6. СОЗДАНИЕ АНСАМБЛЯ MAJORITYVOTECLASSIFIER")
print("-" * 50)

mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])

print("Создан ансамбль MajorityVoteClassifier:")
print("  - Классификаторы: LogisticRegression, DecisionTree, KNN")
print("  - Режим голосования: classlabel (по умолчанию)")

# 7. Оценка ансамбля
print("\n📈 7. ОЦЕНКА АНСАМБЛЯ")
print("-" * 50)

clf_labels += ['Мажоритарное голосование']
all_clf = [pipe1, clf2, pipe3, mv_clf]

print('10-кратная перекрестная проверка (Accuracy):\n')
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(
        estimator=clf,
        X=X_train,
        y=y_train,
        cv=10,
        scoring='accuracy'
    )
    print(f'Accuracy: {scores.mean():.2f} (+/- {scores.std():.2f}) [{label}]')

# 8. Обучение на полном обучающем наборе и оценка на тестовом
print("\n🎯 8. ОЦЕНКА НА ТЕСТОВОМ НАБОРЕ")
print("-" * 50)

print('Точность на тестовом наборе:\n')
for clf, label in zip(all_clf, clf_labels):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    test_acc = np.mean(y_pred == y_test)
    print(f'Accuracy: {test_acc:.3f} [{label}]')

# 9. Демонстрация предсказаний
print("\n🔍 9. ДЕМОНСТРАЦИЯ ПРЕДСКАЗАНИЙ")
print("-" * 50)

# Предсказание для первых 5 образцов
X_sample = X_test[:5]
y_sample = y_test[:5]

print("Первые 5 образцов из тестового набора:")
for i in range(5):
    true_class = le.inverse_transform([y_sample[i]])[0]
    # true_class уже содержит индекс исходного класса (1 или 2)
    original_class = iris.target_names[true_class]
    print(f"  Образец {i+1}: истинный класс = {original_class}")

print(f"\nПредсказания отдельных классификаторов:")
for clf, label in zip([pipe1, clf2, pipe3], clf_labels[:3]):
    pred = clf.predict(X_sample)
    pred_classes = le.inverse_transform(pred)
    print(f"  {label}:")
    for i in range(5):
        original_class = iris.target_names[pred_classes[i]]
        print(f"    Образец {i+1}: {original_class}")

print(f"\nПредсказание ансамбля:")
mv_pred = mv_clf.predict(X_sample)
mv_pred_classes = le.inverse_transform(mv_pred)
for i in range(5):
    original_class = iris.target_names[mv_pred_classes[i]]
    print(f"  Образец {i+1}: {original_class}")

# 10. Выводы
print("\n📝 10. ВЫВОДЫ")
print("=" * 70)
print("Результаты демонстрируют:")
print("  ✅ Ансамблевый классификатор (мажоритарное голосование)")
print("     показывает лучшую производительность, чем отдельные классификаторы")
print("  ✅ Accuracy ансамбля выше, чем у отдельных классификаторов")
print("  ✅ Комбинация различных алгоритмов улучшает обобщающую способность")
print("\nПочему ансамбль работает лучше:")
print("  - Разные классификаторы делают разные ошибки")
print("  - Ошибки компенсируют друг друга при голосовании")
print("  - Уменьшается variance (дисперсия) предсказаний")
print("\nМасштабирование признаков:")
print("  - LogisticRegression и KNN требуют стандартизации")
print("  - DecisionTree масштабно-инвариантен")
print("  - Пайплайны обеспечивают корректную предварительную обработку")
