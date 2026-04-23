# -*- coding: utf-8 -*-
"""
Раздел 6.2.2: Стратифицированная k-кратная перекрёстная проверка

Некоторое улучшение по сравнению со стандартным подходом k-кратной перекрестной
проверки дает стратифицированная k-кратная перекрестная проверка, которая обеспечивает
более точные оценки смещения и дисперсии, особенно в случаях неравных долей классов.

При стратифицированной перекрестной проверке в каждой подвыборке тщательно сохраняют
соотношение классов, чтобы гарантировать, что все подвыборки одинаково представляют
пропорции классов в обучающем наборе данных.
"""
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline

print("📊 СТРАТИФИЦИРОВАННАЯ K-КРАТНАЯ ПЕРЕКРЁСТНАЯ ПРОВЕРКА")
print("=" * 60)

# 1. Загрузка набора данных Wine
print("\n📂 1. ЗАГРУЗКА ДАННЫХ WINE")
print("-" * 40)

df_wine = pd.read_csv('wine.data', header=None)
df_wine.columns = ['Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash',
                   'Magnesium', 'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315', 'Proline']

X = df_wine.iloc[:, 1:].values
y = df_wine.iloc[:, 0].values

print(f'Классы: {np.unique(y)}')
print(f'Форма X: {X.shape}')
print(f'Форма y: {y.shape}')
print(f'\nРаспределение классов:')
for cls in np.unique(y):
    count = np.sum(y == cls)
    print(f'  Класс {cls}: {count} образцов')

# 2. Разделение на обучающие и тестовые наборы
print("\n🔀 2. РАЗДЕЛЕНИЕ ДАННЫХ")
print("-" * 40)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1, stratify=y
)

print(f'Обучающий набор: {X_train.shape[0]} образцов')
print(f'Тестовый набор: {X_test.shape[0]} образцов')
print(f'Распределение в обучающем наборе:')
for cls in np.unique(y_train):
    count = np.sum(y_train == cls)
    print(f'  Класс {cls}: {count} образцов')

# 3. Создание конвейера (Pipeline)
print("\n🔧 3. СОЗДАНИЕ КОНВЕЙЕРА")
print("-" * 40)

# Конвейер: стандартизация + логистическая регрессия
pipe_lr = make_pipeline(StandardScaler(), LogisticRegression(C=100.0, solver='lbfgs', random_state=1))

print("Конвейер включает:")
print("  1. StandardScaler - стандартизация признаков")
print("  2. LogisticRegression - логистическая регрессия (C=100.0)")
print("\nПреимущество конвейера: масштабирование выполняется внутри каждой")
print("складки перекрёстной проверки, что предотвращает утечку данных.")

# 4. Стратифицированная k-кратная перекрёстная проверка (вручную)
print("\n🔄 4. СТРАТИФИЦИРОВАННАЯ K-КРАТНАЯ ПРОВЕРКА (ВРУЧНУЮ)")
print("-" * 40)

kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)

scores = []
for k, (train_idx, test_idx) in enumerate(kfold.split(X_train, y_train)):
    # Обучение модели на обучающей складке
    pipe_lr.fit(X_train[train_idx], y_train[train_idx])
    
    # Оценка на тестовой складке
    score = pipe_lr.score(X_train[test_idx], y_train[test_idx])
    scores.append(score)
    
    # Распределение классов в обучающей части складки
    unique, counts = np.unique(y_train[train_idx], return_counts=True)
    class_dist_str = ', '.join([f'{u}:{c}' for u, c in zip(unique, counts)])
    
    print(f'Выборка: {k+1:02d}, '
          f'Распр. кл.: [{class_dist_str}], '
          f'Точн.: {score:.3f}')

# Расчёт средней точности и стандартного отклонения
mean_acc = np.mean(scores)
std_acc = np.std(scores)

print(f'\n📈 Точность по CV: {mean_acc:.3f} +/- {std_acc:.3f}')

# 5. Стратифицированная k-кратная проверка с cross_val_score
print("\n🚀 5. СТРАТИФИЦИРОВАННАЯ K-КРАТНАЯ ПРОВЕРКА (cross_val_score)")
print("-" * 40)

# cross_val_score с явной передачей kfold для идентичных результатов
cv_scores = cross_val_score(
    estimator=pipe_lr,
    X=X_train,
    y=y_train,
    cv=kfold,  # Явно передаём тот же StratifiedKFold
    n_jobs=1  # Можно установить -1 для использования всех ядер CPU
)

print(f'Оценки точности по CV: {cv_scores}')
print(f'\nТочность по CV: {np.mean(cv_scores):.3f} +/- {np.std(cv_scores):.3f}')

# 6. Сравнение результатов
print("\n📊 6. СРАВНЕНИЕ РЕЗУЛЬТАТОВ")
print("-" * 40)

print(f'Ручной StratifiedKFold:')
print(f'  Средняя точность: {mean_acc:.4f}')
print(f'  Стандартное отклонение: {std_acc:.4f}')
print(f'  Минимальная точность: {np.min(scores):.4f}')
print(f'  Максимальная точность: {np.max(scores):.4f}')

print(f'\ncross_val_score:')
print(f'  Средняя точность: {np.mean(cv_scores):.4f}')
print(f'  Стандартное отклонение: {np.std(cv_scores):.4f}')
print(f'  Минимальная точность: {np.min(cv_scores):.4f}')
print(f'  Максимальная точность: {np.max(cv_scores):.4f}')

# Проверка совпадения результатов
results_match = np.allclose(scores, cv_scores)
print(f'\nРезультаты совпадают: {results_match}')

# 7. Визуализация результатов
print("\n📈 7. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("-" * 40)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 7.1 График точности по складкам
ax1 = axes[0]
fold_numbers = range(1, 11)
ax1.plot(fold_numbers, scores, marker='o', markersize=8, linewidth=2, 
         color='blue', label='Ручной StratifiedKFold')
ax1.plot(fold_numbers, cv_scores, marker='s', markersize=8, linewidth=2, 
         color='red', alpha=0.7, label='cross_val_score')
ax1.axhline(y=mean_acc, color='blue', linestyle='--', alpha=0.5, 
            label=f'Средняя точность ({mean_acc:.3f})')
ax1.fill_between(fold_numbers, 
                 mean_acc - std_acc, 
                 mean_acc + std_acc, 
                 alpha=0.2, color='blue',
                 label='±1 стандартное отклонение')
ax1.set_xlabel('Номер складки')
ax1.set_ylabel('Точность')
ax1.set_title('Точность по складкам стратифицированной k-кратной проверки')
ax1.set_xticks(fold_numbers)
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0.85, 1.05)

# 7.2 Boxplot распределения точности
ax2 = axes[1]
data_to_plot = [scores, cv_scores]
bp = ax2.boxplot(data_to_plot, tick_labels=['Ручной\nStratifiedKFold', 'cross_val_score'],
                 patch_artist=True)
# Раскраска боксплотов
colors_box = ['lightblue', 'lightcoral']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)

# Добавление точек данных
for i, data in enumerate(data_to_plot):
    x = np.random.normal(i + 1, 0.04, size=len(data))
    ax2.scatter(x, data, alpha=0.6, color='black', s=50)

ax2.set_ylabel('Точность')
ax2.set_title('Распределение точности по складкам')
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_ylim(0.85, 1.05)

plt.tight_layout()
plt.show()

# 8. Оценка на тестовом наборе
print("\n🎯 8. ФИНАЛЬНАЯ ОЦЕНКА НА ТЕСТОВОМ НАБОРЕ")
print("-" * 40)

pipe_lr.fit(X_train, y_train)
y_pred = pipe_lr.predict(X_test)
test_accuracy = np.mean(y_pred == y_test)

print(f'Точность на тестовом наборе: {test_accuracy:.3f}')
print(f'Ошибок: {(y_pred != y_test).sum()} из {len(y_test)}')

# 9. Выводы
print("\n📝 9. ВЫВОДЫ")
print("=" * 60)
print("Стратифицированная k-кратная перекрёстная проверка:")
print("  ✅ Обеспечивает более точные оценки смещения и дисперсии")
print("  ✅ Сохраняет пропорции классов в каждой складке")
print("  ✅ Особенно полезна при неравных долях классов")
print("  ✅ Конвейер предотвращает утечку данных при масштабировании")
print("\nПреимущества cross_val_score:")
print("  ✅ Меньше кода для написания")
print("  ✅ Поддержка параллельных вычислений (n_jobs=-1)")
print("  ✅ Автоматическое определение стратификации для классификации")
print("\nРекомендация:")
print("  Использовать cross_val_score для быстрой оценки модели,")
print("  а StratifiedKFold вручную - когда нужен больший контроль")
print("  над процессом (например, для кастомных метрик).")
