# -*- coding: utf-8 -*-
"""
Раздел 7.2.3: Оценка и настройка ансамблевого классификатора

Пример демонстрирует:
1. Построение кривых ROC для всех классификаторов
2. Визуализацию областей принятия решений
3. Демонстрацию метода get_params для доступа к параметрам
4. Настройку гиперпараметров с помощью GridSearchCV
"""
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_curve, auc
from itertools import product

# Импорт MajorityVoteClassifier
import importlib.util
spec = importlib.util.spec_from_file_location("majority_vote", "ch7.2.1_majority_vote_classifier.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
MajorityVoteClassifier = module.MajorityVoteClassifier

print("📊 ОЦЕНКА И НАСТРОЙКА АНСАМБЛЕВОГО КЛАССИФИКАТОРА")
print("=" * 70)

# 1. Загрузка и подготовка данных
print("\n📂 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ IRIS")
print("-" * 50)

iris = datasets.load_iris()
X = iris.data[50:, [1, 2]]  # ширина чашелистика и длина лепестка
y = iris.target[50:]

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=1, stratify=y
)

print(f"Обучающий набор: {X_train.shape[0]} образцов")
print(f"Тестовый набор: {X_test.shape[0]} образцов")

# 2. Создание классификаторов
print("\n🔧 2. СОЗДАНИЕ КЛАССИФИКАТОРОВ")
print("-" * 50)

clf1 = LogisticRegression(C=0.001, solver='lbfgs', random_state=1)
clf2 = DecisionTreeClassifier(max_depth=1, criterion='entropy', random_state=0)
clf3 = KNeighborsClassifier(n_neighbors=1, p=2, metric='minkowski')

pipe1 = Pipeline([['sc', StandardScaler()], ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()], ['clf', clf3]])

mv_clf = MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])

clf_labels = ['Логистическая регрессия', 'Дерево решений', 'KNN', 'Мажоритарное голосование']
all_clf = [pipe1, clf2, pipe3, mv_clf]

print("Классификаторы созданы")

# 3. Построение кривых ROC
print("\n📈 3. ПОСТРОЕНИЕ КРИВЫХ ROC")
print("-" * 50)

colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-']

plt.figure(figsize=(8, 6))

for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):
    # Обучение и предсказание вероятностей
    y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_true=y_test, y_score=y_pred)
    roc_auc = auc(x=fpr, y=tpr)
    
    plt.plot(fpr, tpr,
             color=clr,
             linestyle=ls,
             label=f'{label} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1],
         linestyle='--',
         color='gray',
         linewidth=2,
         label='Случайный классификатор')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.grid(alpha=0.3)
plt.xlabel('Доля ложноположительных прогнозов (FPR)', fontsize=12)
plt.ylabel('Доля истинно положительных прогнозов (TPR)', fontsize=12)
plt.title('Кривые ROC для различных классификаторов', fontsize=14, fontweight='bold')
plt.legend(loc='lower right', fontsize=10)
plt.tight_layout()
plt.show()

print("Кривые ROC построены")

# 4. Визуализация областей принятия решений
print("\n🎨 4. ВИЗУАЛИЗАЦИЯ ОБЛАСТЕЙ ПРИНЯТИЯ РЕШЕНИЙ")
print("-" * 50)

# Стандартизация данных для визуализации
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)

# Создание сетки для визуализации
x_min = X_train_std[:, 0].min() - 1
x_max = X_train_std[:, 0].max() + 1
y_min = X_train_std[:, 1].min() - 1
y_max = X_train_std[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(nrows=2, ncols=2,
                        sharex='col',
                        sharey='row',
                        figsize=(10, 8))

for idx, clf, tt in zip(product([0, 1], [0, 1]), all_clf, clf_labels):
    clf.fit(X_train_std, y_train)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 0, 0],
                                  X_train_std[y_train == 0, 1],
                                  c='blue', marker='^', s=50, label='Versicolor')
    axarr[idx[0], idx[1]].scatter(X_train_std[y_train == 1, 0],
                                  X_train_std[y_train == 1, 1],
                                  c='green', marker='o', s=50, label='Virginica')
    axarr[idx[0], idx[1]].set_title(tt, fontsize=11, fontweight='bold')
    axarr[idx[0], idx[1]].set_xlabel('Ширина чашелистика [стандартизована]', fontsize=9)
    axarr[idx[0], idx[1]].set_ylabel('Длина лепестка [стандартизована]', fontsize=9)
    axarr[idx[0], idx[1]].legend(loc='best', fontsize=8)
    axarr[idx[0], idx[1]].grid(alpha=0.3)

plt.suptitle('Области принятия решений классификаторов', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("Области принятия решений визуализированы")

# 5. Демонстрация метода get_params
print("\n🔍 5. ДЕМОНСТРАЦИЯ МЕТОДА GET_PARAMS")
print("-" * 50)

print("Параметры ансамблевого классификатора:")
params = mv_clf.get_params()
print(f"Всего параметров: {len(params)}")
print("\nОсновные параметры:")
for key in sorted(params.keys())[:15]:
    print(f"  {key}: {params[key]}")
print("  ...")
print("\nПараметры логистической регрессии:")
for key in sorted(params.keys()):
    if 'pipeline-1' in key and 'clf' in key and not key.endswith('__'):
        print(f"  {key}: {params[key]}")

# 6. Настройка гиперпараметров с помощью GridSearchCV
print("\n⚙️ 6. НАСТРОЙКА ГИПЕРПАРАМЕТРОВ С ПОМОЩЬЮ GRIDSEARCHCV")
print("-" * 50)

params = {
    'decisiontreeclassifier__max_depth': [1, 2],
    'pipeline-1__clf__C': [0.001, 0.1, 100.0]
}

print("Сетка параметров для поиска:")
print(f"  decisiontreeclassifier__max_depth: {params['decisiontreeclassifier__max_depth']}")
print(f"  pipeline-1__clf__C: {params['pipeline-1__clf__C']}")

grid = GridSearchCV(estimator=mv_clf,
                    param_grid=params,
                    cv=10,
                    scoring='accuracy')

print("\nЗапуск GridSearchCV...")
grid.fit(X_train, y_train)

print("\nРезультаты GridSearchCV:")
print("-" * 50)
for r, _ in enumerate(grid.cv_results_['mean_test_score']):
    mean_score = grid.cv_results_['mean_test_score'][r]
    std_dev = grid.cv_results_['std_test_score'][r]
    params = grid.cv_results_['params'][r]
    print(f'{mean_score:.3f} +/- {std_dev:.2f} {params}')

print(f"\nЛучшие параметры: {grid.best_params_}")
print(f'Лучшая точность: {grid.best_score_:.3f}')

# 7. Оценка лучшей модели на тестовом наборе
print("\n🎯 7. ОЦЕНКА ЛУЧШЕЙ МОДЕЛИ НА ТЕСТОВОМ НАБОРЕ")
print("-" * 50)

best_clf = grid.best_estimator_
best_clf.fit(X_train, y_train)
y_pred = best_clf.predict(X_test)
test_acc = np.mean(y_pred == y_test)

print(f"Точность на тестовом наборе: {test_acc:.3f}")

# 8. Выводы
print("\n📝 8. ВЫВОДЫ")
print("=" * 70)
print("Результаты демонстрируют:")
print("  ✅ Кривые ROC показывают производительность классификаторов")
print("  ✅ Ансамбль показывает высокую производительность (AUC)")
print("  ✅ Области принятия решений ансамбля - гибрид отдельных классификаторов")
print("  ✅ Метод get_params позволяет доступ к параметрам вложенных классификаторов")
print("  ✅ GridSearchCV эффективно настраивает гиперпараметры ансамбля")
print("\nНаблюдения:")
print("  - Ансамбль объединяет сильные стороны отдельных классификаторов")
print("  - Логистическая регрессия с сильной регуляризацией (C=0.001) работает лучше")
print("  - Глубина дерева решений (1 или 2) мало влияет на результат")
print("  - Области решений ансамбля похожи на дерево решений, но с нелинейностью от KNN")
