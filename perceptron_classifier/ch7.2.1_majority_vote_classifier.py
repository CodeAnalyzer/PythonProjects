# -*- coding: utf-8 -*-
"""
Раздел 7.2.1: Реализация простого мажоритарного классификатора

MajorityVoteClassifier - ансамблевый классификатор, который комбинирует
различные алгоритмы классификации с использованием мажоритарного голосования.

Два режима работы:
1. vote='classlabel': голосование на основе предсказанных меток классов
2. vote='probability': голосование на основе усреднённых вероятностей классов

Преимущества:
- Комбинация различных алгоритмов для улучшения обобщающей способности
- Возможность взвешивания классификаторов
- Совместимость с GridSearchCV через get_params/set_params

Принцип работы:
- Каждый классификатор делает предсказание
- Результаты комбинируются (мажоритарное голосование или усреднение вероятностей)
- Ансамбль возвращает финальное предсказание
"""
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import _name_estimators
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

print("🗳️ МАЖОРИТАРНЫЙ КЛАССИФИКАТОР (MAJORITYVOTECLASSIFIER)")
print("=" * 60)

# 1. Реализация класса MajorityVoteClassifier
print("\n🔧 1. РЕАЛИЗАЦИЯ КЛАССА MAJORITYVOTECLASSIFIER")
print("-" * 40)

class MajorityVoteClassifier(BaseEstimator, ClassifierMixin):
    """
    Ансамблевый классификатор на основе мажоритарного голосования.
    
    Параметры:
    -----------
    classifiers : list
        Список классификаторов для ансамбля
    vote : str, default='classlabel'
        Режим голосования: 'classlabel' или 'probability'
    weights : array-like, shape=[n_classifiers], optional
        Веса для классификаторов
    """
    
    # Тег для распознавания sklearn как классификатора
    _estimator_type = 'classifier'
    
    def __init__(self, classifiers, vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.named_classifiers = {key: value for key, value in _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights
    
    def fit(self, X, y):
        """
        Обучение всех классификаторов.
        
        Параметры:
        -----------
        X : array-like, shape=[n_samples, n_features]
            Матрица признаков
        y : array-like, shape=[n_samples]
            Вектор меток классов
        """
        # Проверка параметра vote
        if self.vote not in ('probability', 'classlabel'):
            raise ValueError(f"vote должен быть 'probability' или 'classlabel'; "
                           f"получено vote={self.vote}")
        
        # Проверка соответствия весов
        if self.weights and len(self.weights) != len(self.classifiers):
            raise ValueError(f'Количество классификаторов и весов должно совпадать; '
                           f'имеется {len(self.weights)} весов и {len(self.classifiers)} классификаторов')
        
        # LabelEncoder для проверки, что метки начинаются с 0
        self.labelenc_ = LabelEncoder()
        self.labelenc_.fit(y)
        self.classes_ = self.labelenc_.classes_
        
        # Обучение каждого классификатора
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, self.labelenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        
        return self
    
    def predict(self, X):
        """
        Предсказание метки класса.
        
        Параметры:
        -----------
        X : array-like, shape=[n_samples, n_features]
            Матрица признаков
        
        Возвращает:
        -----------
        maj_vote : array-like, shape=[n_samples]
            Предсказанные метки классов
        """
        if self.vote == 'probability':
            # Голосование на основе вероятностей
            maj_vote = np.argmax(self.predict_proba(X), axis=1)
        else:
            # Голосование на основе меток классов
            predictions = np.asarray([clf.predict(X) for clf in self.classifiers_]).T
            
            # Мажоритарное голосование с весами
            maj_vote = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self.weights)),
                axis=1,
                arr=predictions
            )
        
        # Обратное преобразование меток
        maj_vote = self.labelenc_.inverse_transform(maj_vote)
        return maj_vote
    
    def predict_proba(self, X):
        """
        Предсказание вероятностей классов.
        
        Параметры:
        -----------
        X : array-like, shape=[n_samples, n_features]
            Матрица признаков
        
        Возвращает:
        -----------
        avg_proba : array-like, shape=[n_samples, n_classes]
            Усреднённые вероятности классов
        """
        probas = np.asarray([clf.predict_proba(X) for clf in self.classifiers_])
        avg_proba = np.average(probas, axis=0, weights=self.weights)
        return avg_proba
    
    def get_params(self, deep=True):
        """
        Получение параметров классификатора для GridSearchCV.
        """
        if not deep:
            return super().get_params(deep=False)
        else:
            out = self.named_classifiers.copy()
            for name, step in self.named_classifiers.items():
                for key, value in step.get_params(deep=True).items():
                    out[f'{name}__{key}'] = value
            return out

print("Класс MajorityVoteClassifier реализован:")
print("  - Наследуется от BaseEstimator и ClassifierMixin")
print("  - Поддерживает режимы 'classlabel' и 'probability'")
print("  - Совместим с GridSearchCV через get_params")

# 2. Демонстрация np.argmax и np.bincount
print("\n📊 2. ДЕМОНСТРАЦИЯ np.ARGMAX И np.BINCOUNT")
print("-" * 40)

# Пример с bincount и argmax
labels = [0, 0, 1]
weights = [0.2, 0.2, 0.6]
result = np.argmax(np.bincount(labels, weights=weights))

print("Пример 1: Голосование на основе меток")
print(f"  Метки: {labels}")
print(f"  Веса: {weights}")
print(f"  bincount с весами: {np.bincount(labels, weights=weights)}")
print(f"  argmax (индекс максимального значения): {result}")
print(f"  Победивший класс: {result}")

# Пример с вероятностями
ex = np.array([[0.9, 0.1],
               [0.8, 0.2],
               [0.4, 0.6]])
p = np.average(ex, axis=0, weights=[0.2, 0.2, 0.6])
result_proba = np.argmax(p)

print(f"\nПример 2: Голосование на основе вероятностей")
print(f"  Вероятности классификаторов:")
print(f"    Классификатор 1: {ex[0]}")
print(f"    Классификатор 2: {ex[1]}")
print(f"    Классификатор 3: {ex[2]}")
print(f"  Веса: {weights}")
print(f"  Средние вероятности: {p}")
print(f"  argmax (индекс максимальной вероятности): {result_proba}")
print(f"  Победивший класс: {result_proba}")

# 3. Загрузка данных
print("\n📂 3. ЗАГРУЗКА ДАННЫХ WDBC")
print("-" * 40)

df_wdbc = pd.read_csv('wdbc.data', header=None)
df_wdbc.columns = ['ID', 'Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]

# Кодирование диагноза
le = LabelEncoder()
df_wdbc['Diagnosis'] = le.fit_transform(df_wdbc['Diagnosis'])

X = df_wdbc.iloc[:, 2:].values
y = df_wdbc['Diagnosis'].values

print(f'Классы: {np.unique(y)}')
print(f'Классовые метки: {le.classes_}')
print(f'Форма X: {X.shape}')
print(f'Форма y: {y.shape}')

# Разделение на обучающие и тестовые наборы
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

print(f'Обучающий набор: {X_train.shape[0]} образцов')
print(f'Тестовый набор: {X_test.shape[0]} образцов')

# 4. Создание базовых классификаторов
print("\n🔧 4. СОЗДАНИЕ БАЗОВЫХ КЛАССИФИКАТОРОВ")
print("-" * 40)

# Logistic Regression
clf1 = LogisticRegression(C=0.1, random_state=1, solver='lbfgs', max_iter=10000)

# Decision Tree
clf2 = DecisionTreeClassifier(max_depth=3, criterion='entropy', random_state=0)

# KNN
clf3 = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')

print("Базовые классификаторы:")
print("  1. LogisticRegression (C=0.1)")
print("  2. DecisionTreeClassifier (max_depth=3)")
print("  3. KNeighborsClassifier (n_neighbors=5)")

# 5. Оценка отдельных классификаторов
print("\n📊 5. ОЦЕНКА ОТДЕЛЬНЫХ КЛАССИФИКАТОРОВ")
print("-" * 40)

# Конвейеры для каждого классификатора
pipe1 = make_pipeline(StandardScaler(), clf1)
pipe2 = make_pipeline(StandardScaler(), clf2)
pipe3 = make_pipeline(StandardScaler(), clf3)

clf_labels = ['Logistic Regression', 'Decision Tree', 'KNN']

print('Перекрёстная проверка (10-fold):')
for clf, label in zip([pipe1, pipe2, pipe3], clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, n_jobs=-1)
    print(f'  {label:<25} Accuracy: {scores.mean():.3f} +/- {scores.std():.3f}')

# 6. Создание ансамбля MajorityVoteClassifier
print("\n🗳️ 6. СОЗДАНИЕ АНСАМБЛЯ MAJORITYVOTECLASSIFIER")
print("-" * 40)

# Ансамбль с голосованием по меткам
mv_clf = MajorityVoteClassifier(classifiers=[pipe1, pipe2, pipe3])

print("Ансамбль MajorityVoteClassifier:")
print("  - Классификаторы: Logistic Regression, Decision Tree, KNN")
print("  - Режим голосования: classlabel (по умолчанию)")
print("  - Веса: None (равные веса)")

# Оценка ансамбля
clf_labels += ['Majority Voting']
all_clf = [pipe1, pipe2, pipe3, mv_clf]

print('\nПерекрёстная проверка всех классификаторов:')
for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf, X=X_train, y=y_train, cv=10, n_jobs=-1)
    print(f'  {label:<25} Accuracy: {scores.mean():.3f} +/- {scores.std():.3f}')

# 7. Сравнение на тестовом наборе
print("\n📊 7. СРАВНЕНИЕ НА ТЕСТОВОМ НАБОРЕ")
print("-" * 40)

print('Точность на тестовом наборе:')
for clf, label in zip(all_clf, clf_labels):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    test_acc = np.mean(y_pred == y_test)
    print(f'  {label:<25} Accuracy: {test_acc:.3f}')

# 8. Ансамбль с взвешенным голосованием
print("\n⚖️ 8. АНСАМБЛЬ СО ВЗВЕШЕННЫМ ГОЛОСОВАНИЕМ")
print("-" * 40)

# Веса: LogisticRegression имеет больший вес
weights = [0.4, 0.3, 0.3]
mv_clf_weighted = MajorityVoteClassifier(classifiers=[pipe1, pipe2, pipe3], weights=weights)

print(f"Ансамбль с весами: {weights}")
print("  - LogisticRegression: 0.4")
print("  - Decision Tree: 0.3")
print("  - KNN: 0.3")

mv_clf_weighted.fit(X_train, y_train)
y_pred_weighted = mv_clf_weighted.predict(X_test)
test_acc_weighted = np.mean(y_pred_weighted == y_test)

print(f'\nТочность на тестовом наборе: {test_acc_weighted:.3f}')

# 9. Ансамбль с голосованием по вероятностям
print("\n📈 9. АНСАМБЛЬ С ГОЛОСОВАНИЕМ ПО ВЕРОЯТНОСТЯМ")
print("-" * 40)

mv_clf_proba = MajorityVoteClassifier(classifiers=[pipe1, pipe2, pipe3], vote='probability')

print("Ансамбль с vote='probability':")
print("  - Голосование на основе усреднённых вероятностей")
print("  - Требует, чтобы все классификаторы поддерживали predict_proba")

mv_clf_proba.fit(X_train, y_train)
y_pred_proba = mv_clf_proba.predict(X_test)
test_acc_proba = np.mean(y_pred_proba == y_test)

print(f'\nТочность на тестовом наборе: {test_acc_proba:.3f}')

# 10. Сравнение всех подходов
print("\n📊 10. СРАВНЕНИЕ ВСЕХ ПОДХОДОВ")
print("-" * 40)

approaches = [
    'Logistic Regression',
    'Decision Tree',
    'KNN',
    'Majority Voting (equal weights)',
    'Majority Voting (weighted)',
    'Majority Voting (probability)'
]

test_accuracies = [
    test_acc,  # Logistic Regression (будет вычислено ниже)
    test_acc,  # Decision Tree (будет вычислено ниже)
    test_acc,  # KNN (будет вычислено ниже)
    test_acc,  # Majority Voting (будет вычислено ниже)
    test_acc_weighted,
    test_acc_proba
]

# Пересчитаем точности для отдельных классификаторов
pipe1.fit(X_train, y_train)
pipe2.fit(X_train, y_train)
pipe3.fit(X_train, y_train)
test_accuracies[0] = np.mean(pipe1.predict(X_test) == y_test)
test_accuracies[1] = np.mean(pipe2.predict(X_test) == y_test)
test_accuracies[2] = np.mean(pipe3.predict(X_test) == y_test)
mv_clf.fit(X_train, y_train)
test_accuracies[3] = np.mean(mv_clf.predict(X_test) == y_test)

print(f"{'Подход':<40} {'Точность':<10}")
print('-' * 55)
for approach, acc in zip(approaches, test_accuracies):
    print(f'{approach:<40} {acc:<10.3f}')

# 11. Визуализация результатов
print("\n📈 11. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("-" * 40)

plt.figure(figsize=(12, 6))
bars = plt.bar(approaches, test_accuracies, color=['lightblue', 'lightgreen', 'lightcoral', 
                                                  'gold', 'orange', 'purple'], 
               alpha=0.7, edgecolor='black')
plt.ylabel('Точность', fontsize=12)
plt.title('Сравнение точности различных подходов', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
plt.ylim([0.5, 1.0])

# Добавление значений на столбцы
for bar, acc in zip(bars, test_accuracies):
    plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
             f'{acc:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# 12. Демонстрация predict_proba
print("\n📊 12. ДЕМОНСТРАЦИЯ PREDICT_PROBA")
print("-" * 40)

# Предсказание первых 5 образцов
X_sample = X_test[:5]
y_sample = y_test[:5]

print("Первые 5 образцов из тестового набора:")
for i in range(5):
    true_label = le.inverse_transform([y_sample[i]])[0]
    print(f"  Образец {i+1}: истинный класс = {true_label}")

print(f"\nВероятности классов для ансамбля (vote='probability'):")
probas = mv_clf_proba.predict_proba(X_sample)
for i in range(5):
    print(f"  Образец {i+1}: {probas[i]}")
    pred = mv_clf_proba.predict(X_sample[i:i+1])[0]
    pred_label = le.inverse_transform([pred])[0]
    print(f"    Предсказанный класс: {pred_label}")

# 13. Выводы
print("\n📝 13. ВЫВОДЫ")
print("=" * 60)
print("MajorityVoteClassifier позволяет:")
print("  ✅ Комбинировать различные алгоритмы классификации")
print("  ✅ Улучшать точность за счёт мажоритарного голосования")
print("  ✅ Использовать взвешивание классификаторов")
print("  ✅ Выбирать режим голосования (метки или вероятности)")
print("  ✅ Интегрироваться с GridSearchCV")
print("\nРежимы голосования:")
print("  - vote='classlabel': мажоритарное голосование по меткам")
print("  - vote='probability': усреднение вероятностей классов")
print("\nВзвешивание:")
print("  - Можно назначить разные веса классификаторам")
print("  - Полезно, когда один классификатор надёжнее других")
print("\nСовместимость с sklearn:")
print("  - Наследование от BaseEstimator и ClassifierMixin")
print("  - Методы get_params и set_params для GridSearchCV")
print("  - Метод score для оценки точности")
print("\nПрактическое применение:")
print("  - Комбинация разных алгоритмов для улучшения результатов")
print("  - Ансамблирование моделей с разными гиперпараметрами")
print("  - Уменьшение переобучения и улучшение обобщения")
