# -*- coding: utf-8 -*-
"""
Раздел 6.4.4: Выбор алгоритма методом вложенной перекрестной проверки

Вложенная перекрестная проверка (Nested Cross-Validation) - это метод для
сравнения различных алгоритмов машинного обучения с несмещённой оценкой ошибки.

Структура вложенной перекрестной проверки:
- Внешний цикл k-кратной CV: разделение данных на обучающую и тестовую выборки
- Внутренний цикл k-кратной CV: выбор модели (подбор гиперпараметров) на обучающей выборке
- Тестовая выборка: оценка производительности выбранной модели

Преимущества:
- Несмещённая оценка ошибки (почти не смещена по сравнению с тестовым набором)
- Честное сравнение различных алгоритмов
- Предотвращение утечки данных при подборе гиперпараметров

Пример: 5×2 перекрёстная проверка (5 внешних складок, 2 внутренних)
"""
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

print("🔄 ВЛОЖЕННАЯ ПЕРЕКРЁСТНАЯ ПРОВЕРКА: ВЫБОР АЛГОРИТМА")
print("=" * 60)

# 1. Загрузка набора данных WDBC
print("\n📂 1. ЗАГРУЗКА ДАННЫХ WDBC")
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

# 2. Разделение на обучающие и тестовые наборы
print("\n🔀 2. РАЗДЕЛЕНИЕ ДАННЫХ")
print("-" * 40)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

print(f'Обучающий набор: {X_train.shape[0]} образцов')
print(f'Тестовый набор: {X_test.shape[0]} образцов')

# 3. Вложенная перекрёстная проверка для SVM
print("\n🎯 3. ВЛОЖЕННАЯ CV ДЛЯ SVM")
print("-" * 40)

# Создание конвейера для SVM
pipe_svc = make_pipeline(
    StandardScaler(),
    SVC(random_state=1)
)

# Сетка гиперпараметров для SVM
param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid_svc = [
    {
        'svc__C': param_range,
        'svc__kernel': ['linear']
    },
    {
        'svc__C': param_range,
        'svc__gamma': param_range,
        'svc__kernel': ['rbf']
    }
]

print("Сетка гиперпараметров для SVM:")
print("  Конфигурация 1 (линейное ядро):")
print(f"    - svc__C: {param_range}")
print("    - svc__kernel: ['linear']")
print("  Конфигурация 2 (RBF ядро):")
print(f"    - svc__C: {param_range}")
print(f"    - svc__gamma: {param_range}")
print("    - svc__kernel: ['rbf']")

# Внутренний цикл: GridSearchCV для SVM
gs_svc = GridSearchCV(
    estimator=pipe_svc,
    param_grid=param_grid_svc,
    scoring='accuracy',
    cv=2
)

print("\nВнутренний цикл (GridSearchCV):")
print("  - cv=2 (2-кратная перекрёстная проверка для подбора гиперпараметров)")
print("  - scoring='accuracy'")

# Внешний цикл: cross_val_score для SVM
print("\nВнешний цикл (cross_val_score):")
print("  - cv=5 (5-кратная перекрёстная проверка для оценки модели)")
print("  - Это схема 5×2 вложенной перекрёстной проверки")

scores_svc = cross_val_score(gs_svc, X_train, y_train, scoring='accuracy', cv=5)

print(f'\nРезультаты SVM:')
print(f'  Точность перекрёстной проверки: {np.mean(scores_svc):.3f} +/- {np.std(scores_svc):.3f}')
print(f'  Индивидуальные оценки: {scores_svc}')

# 4. Вложенная перекрёстная проверка для Decision Tree
print("\n🌳 4. ВЛОЖЕННАЯ CV ДЛЯ DECISION TREE")
print("-" * 40)

# Сетка гиперпараметров для Decision Tree
param_grid_tree = [
    {
        'max_depth': [1, 2, 3, 4, 5, 6, 7, None]
    }
]

print("Сетка гиперпараметров для Decision Tree:")
print("  - max_depth: [1, 2, 3, 4, 5, 6, 7, None]")
print("    (None означает неограниченную глубину)")

# Внутренний цикл: GridSearchCV для Decision Tree
gs_tree = GridSearchCV(
    estimator=DecisionTreeClassifier(random_state=0),
    param_grid=param_grid_tree,
    scoring='accuracy',
    cv=2
)

print("\nВнутренний цикл (GridSearchCV):")
print("  - cv=2 (2-кратная перекрёстная проверка для подбора гиперпараметров)")

# Внешний цикл: cross_val_score для Decision Tree
print("\nВнешний цикл (cross_val_score):")
print("  - cv=5 (5-кратная перекрёстная проверка для оценки модели)")

scores_tree = cross_val_score(gs_tree, X_train, y_train, scoring='accuracy', cv=5)

print(f'\nРезультаты Decision Tree:')
print(f'  Точность перекрёстной проверки: {np.mean(scores_tree):.3f} +/- {np.std(scores_tree):.3f}')
print(f'  Индивидуальные оценки: {scores_tree}')

# 5. Сравнение алгоритмов
print("\n📊 5. СРАВНЕНИЕ АЛГОРИТМОВ")
print("-" * 40)

print(f'SVM:')
print(f'  Средняя точность: {np.mean(scores_svc):.3f}')
print(f'  Стандартное отклонение: {np.std(scores_svc):.3f}')
print(f'  Минимальная точность: {np.min(scores_svc):.3f}')
print(f'  Максимальная точность: {np.max(scores_svc):.3f}')

print(f'\nDecision Tree:')
print(f'  Средняя точность: {np.mean(scores_tree):.3f}')
print(f'  Стандартное отклонение: {np.std(scores_tree):.3f}')
print(f'  Минимальная точность: {np.min(scores_tree):.3f}')
print(f'  Максимальная точность: {np.max(scores_tree):.3f}')

# Статистическая значимость разницы
diff = np.mean(scores_svc) - np.mean(scores_tree)
print(f'\nРазница в точности: {diff:.3f}')

if diff > 0.05:
    print('✅ SVM значительно лучше Decision Tree')
elif diff > 0.02:
    print('⚠️  SVM умеренно лучше Decision Tree')
elif diff > -0.02:
    print('✅ Алгоритмы сопоставимы')
elif diff > -0.05:
    print('⚠️  Decision Tree умеренно лучше SVM')
else:
    print('✅ Decision Tree значительно лучше SVM')

# 6. Визуализация результатов
print("\n📈 6. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("-" * 40)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# График 1: Сравнение точности по внешним складкам
ax1 = axes[0]
folds = range(1, 6)
ax1.plot(folds, scores_svc, marker='o', markersize=10, linewidth=2, 
         color='blue', label='SVM')
ax1.plot(folds, scores_tree, marker='s', markersize=10, linewidth=2, 
         color='green', label='Decision Tree')
ax1.axhline(y=np.mean(scores_svc), color='blue', linestyle='--', alpha=0.5,
            label=f'SVM средняя: {np.mean(scores_svc):.3f}')
ax1.axhline(y=np.mean(scores_tree), color='green', linestyle='--', alpha=0.5,
            label=f'DT средняя: {np.mean(scores_tree):.3f}')
ax1.set_xlabel('Номер внешней складки')
ax1.set_ylabel('Точность')
ax1.set_title('Сравнение точности по внешним складкам (5×2 CV)')
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.85, 1.0])

# График 2: Boxplot распределения точности
ax2 = axes[1]
data_to_plot = [scores_svc, scores_tree]
bp = ax2.boxplot(data_to_plot, tick_labels=['SVM', 'Decision Tree'],
                 patch_artist=True)
colors_box = ['lightblue', 'lightgreen']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)

# Добавление точек данных
for i, data in enumerate(data_to_plot):
    x = np.random.normal(i + 1, 0.04, size=len(data))
    ax2.scatter(x, data, alpha=0.6, color='black', s=50)

ax2.set_ylabel('Точность')
ax2.set_title('Распределение точности по внешним складкам')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim([0.85, 1.0])

plt.tight_layout()
plt.show()

# 7. Обучение лучших моделей на полном обучающем наборе
print("\n🎯 7. ОБУЧЕНИЕ ЛУЧШИХ МОДЕЛЕЙ НА ПОЛНОМ НАБОРЕ")
print("-" * 40)

# SVM
print("\nSVM:")
gs_svc.fit(X_train, y_train)
print(f'  Лучшие параметры: {gs_svc.best_params_}')
print(f'  Лучшая точность (CV): {gs_svc.best_score_:.4f}')
svm_test_score = gs_svc.score(X_test, y_test)
print(f'  Точность на тестовом наборе: {svm_test_score:.4f}')

# Decision Tree
print("\nDecision Tree:")
gs_tree.fit(X_train, y_train)
print(f'  Лучшие параметры: {gs_tree.best_params_}')
print(f'  Лучшая точность (CV): {gs_tree.best_score_:.4f}')
tree_test_score = gs_tree.score(X_test, y_test)
print(f'  Точность на тестовом наборе: {tree_test_score:.4f}')

# 8. Объяснение вложенной перекрёстной проверки
print("\n📝 8. ОБЪЯСНЕНИЕ ВЛОЖЕННОЙ ПЕРЕКРЁСТНОЙ ПРОВЕРКИ")
print("-" * 40)

print("Структура вложенной перекрёстной проверки (5×2):")
print("  Внешний цикл (5 складок):")
print("    - Разделяет данные на 5 частей")
print("    - 4 части для обучения, 1 часть для тестирования")
print("    - Повторяется 5 раз с разными тестовыми частями")
print("\n  Внутренний цикл (2 складки) на каждой внешней обучающей части:")
print("    - GridSearchCV подбирает гиперпараметры")
print("    - Использует 2-кратную CV для оценки каждой конфигурации")
print("    - Выбирает лучшую конфигурацию")
print("\n  Оценка:")
print("    - Выбранная модель тестируется на внешней тестовой части")
print("    - Повторяется для каждой из 5 внешних складок")
print("    - Усредняется по всем 5 внешним складкам")

print("\nПреимущества:")
print("  ✅ Несмещённая оценка ошибки (почти как на тестовом наборе)")
print("  ✅ Честное сравнение различных алгоритмов")
print("  ✅ Предотвращение утечки данных при подборе гиперпараметров")
print("  ✅ Учитывает неопределённость в подборе гиперпараметров")

print("\nКогда использовать:")
print("  - При сравнении различных алгоритмов ML")
print("  - Когда важна несмещённая оценка ошибки")
print("  - При ограниченном количестве данных")
print("  - Для публикации результатов в научных работах")

print("\nНедостатки:")
print("  ⚠️  Высокие вычислительные затраты")
print("  ⚠️  Долгое время выполнения")
print("  ⚠️  Сложность интерпретации")

# 9. Практические рекомендации
print("\n💡 9. ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ")
print("-" * 40)

print("Выбор параметров вложенной CV:")
print("  - Внешний cv: обычно 5 или 10 (больше → более точная оценка)")
print("  - Внутренний cv: обычно 3, 5 или 10 (больше → лучший подбор гиперпараметров)")
print("  - Компромисс: 5×2 или 5×5 (баланс точности и скорости)")

print("\nАльтернативы:")
print("  - Обычная k-кратная CV + отдельный тестовый набор")
print("  - Stratified k-fold для несбалансированных данных")
print("  - Repeated k-fold для большей стабильности")

print("\nИнтерпретация результатов:")
print("  - Сравнивайте средние значения, а не отдельные оценки")
print("  - Учитывайте стандартное отклонение (меньше → стабильнее)")
print("  - Разница > 0.05 обычно считается значимой")
print("  - Используйте статистические тесты для строгого сравнения")

# 10. Выводы
print("\n📝 10. ВЫВОДЫ")
print("=" * 60)
print("Вложенная перекрёстная проверка позволяет:")
print("  ✅ Честно сравнивать различные алгоритмы машинного обучения")
print("  ✅ Получать несмещённую оценку ошибки модели")
print("  ✅ Учитывать неопределённость в подборе гиперпараметров")
print("  ✅ Предотвращать утечку данных при настройке моделей")
print("\nВ данном примере:")
print(f"  - SVM показал точность {np.mean(scores_svc):.3f} +/- {np.std(scores_svc):.3f}")
print(f"  - Decision Tree показал точность {np.mean(scores_tree):.3f} +/- {np.std(scores_tree):.3f}")
if np.mean(scores_svc) > np.mean(scores_tree):
    print("  - SVM показал лучшие результаты")
    print("  - Рекомендуется использовать SVM для классификации WDBC")
else:
    print("  - Decision Tree показал лучшие результаты")
    print("  - Рекомендуется использовать Decision Tree для классификации WDBC")
print("\nВложенная CV особенно важна при:")
print("  - Научных исследованиях и публикациях")
print("  - Сравнении новых алгоритмов с существующими")
print("  - Ограниченных наборах данных")
print("  - Требовании к высокой достоверности результатов")
