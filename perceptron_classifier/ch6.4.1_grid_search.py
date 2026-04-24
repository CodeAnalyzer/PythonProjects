# -*- coding: utf-8 -*-
"""
Раздел 6.4.1: Настройка гиперпараметров с помощью поиска по сетке

Поиск по сетке (Grid Search) - это метод настройки гиперпараметров, который
заключается в полном переборе всех комбинаций значений гиперпараметров из
заданного списка. Для каждой комбинации оценивается производительность модели
с помощью перекрёстной проверки, и выбирается лучшая комбинация.

Этот метод прост в реализации, но может быть вычислительно затратным при
большом количестве гиперпараметров и их значений.
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
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline

print("🔍 ПОИСК ПО СЕТКЕ: НАСТРОЙКА ГИПЕРПАРАМЕТРОВ SVM")
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

# 3. Создание конвейера
print("\n🔧 3. СОЗДАНИЕ КОНВЕЙЕРА")
print("-" * 40)

pipe_svc = make_pipeline(
    StandardScaler(),
    SVC(random_state=1)
)

print("Конвейер включает:")
print("  1. StandardScaler - стандартизация признаков")
print("  2. SVC - машина опорных векторов")

# 4. Определение сетки гиперпараметров
print("\n📊 4. ОПРЕДЕЛЕНИЕ СЕТКИ ГИПЕРПАРАМЕТРОВ")
print("-" * 40)

param_range = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

param_grid = [
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

print("Сетка гиперпараметров:")
print("  Конфигурация 1 (линейное ядро):")
print(f"    - svc__C: {param_range}")
print(f"    - svc__kernel: ['linear']")
print("  Конфигурация 2 (RBF ядро):")
print(f"    - svc__C: {param_range}")
print(f"    - svc__gamma: {param_range}")
print(f"    - svc__kernel: ['rbf']")

total_combinations = len(param_range) + len(param_range) * len(param_range)
print(f"\nВсего комбинаций для проверки: {total_combinations}")
print(f"С учётом 10-кратной CV: {total_combinations * 10} обучений модели")

# 5. Поиск по сетке
print("\n🔍 5. ПОИСК ПО СЕТКЕ (GridSearchCV)")
print("-" * 40)

gs = GridSearchCV(
    estimator=pipe_svc,
    param_grid=param_grid,
    scoring='accuracy',
    cv=10,
    refit=True,
    n_jobs=-1
)

print("Запуск поиска по сетке...")
print("  - cv=10 (10-кратная перекрёстная проверка)")
print("  - scoring='accuracy' (метрика точности)")
print("  - n_jobs=-1 (использование всех ядер CPU)")
print("  - refit=True (автоматическое обучение лучшей модели)")

gs.fit(X_train, y_train)

print("\n✅ Поиск завершён!")

# 6. Результаты поиска
print("\n📈 6. РЕЗУЛЬТАТЫ ПОИСКА")
print("-" * 40)

print(f'Лучшая точность (CV): {gs.best_score_:.4f}')
print(f'\nЛучшие параметры:')
for param, value in gs.best_params_.items():
    print(f'  {param}: {value}')

# 7. Анализ всех результатов
print("\n📊 7. АНАЛИЗ ВСЕХ РЕЗУЛЬТАТОВ")
print("-" * 40)

results_df = pd.DataFrame(gs.cv_results_)
print(f'Всего проверено комбинаций: {len(results_df)}')

# Топ-5 лучших комбинаций
print('\nТоп-5 лучших комбинаций:')
top_5 = results_df.nlargest(5, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
for idx, row in top_5.iterrows():
    params_str = str(row['params'])
    print(f"  {row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}: {params_str}")

# Сравнение ядер
print("\nСравнение ядер:")
for kernel in ['linear', 'rbf']:
    kernel_results = results_df[results_df['param_svc__kernel'] == kernel]
    if len(kernel_results) > 0:
        best_kernel_score = kernel_results['mean_test_score'].max()
        best_kernel_params = kernel_results.loc[kernel_results['mean_test_score'].idxmax(), 'params']
        print(f"  Ядро '{kernel}': лучшая точность {best_kernel_score:.4f}")
        print(f"    Параметры: {best_kernel_params}")

# 8. Оценка на тестовом наборе
print("\n🎯 8. ФИНАЛЬНАЯ ОЦЕНКА НА ТЕСТОВОМ НАБОРЕ")
print("-" * 40)

# Лучшая модель уже обучена на полном обучающем наборе (refit=True)
clf = gs.best_estimator_
test_score = clf.score(X_test, y_test)

print(f'Точность на тестовом наборе: {test_score:.4f}')
print(f'Ошибок: {(clf.predict(X_test) != y_test).sum()} из {len(y_test)}')

# 9. Сравнение с дефолтными параметрами
print("\n📊 9. СРАВНЕНИЕ С ДЕФОЛТНЫМИ ПАРАМЕТРАМИ")
print("-" * 40)

pipe_svc_default = make_pipeline(
    StandardScaler(),
    SVC(random_state=1)
)

pipe_svc_default.fit(X_train, y_train)
default_score = pipe_svc_default.score(X_test, y_test)

print(f'Дефолтный SVM (C=1.0, kernel="rbf", gamma="scale"):')
print(f'  Точность на тестовом наборе: {default_score:.4f}')

improvement = (test_score - default_score) * 100
if test_score > default_score:
    print(f'\n✅ Оптимизация улучшила точность на {improvement:.2f}%')
elif test_score < default_score:
    degradation = (default_score - test_score) * 100
    print(f'\n⚠️  Дефолтные параметры лучше на {degradation:.2f}%')
else:
    print(f'\n✅ Точность одинакова')

# 10. Визуализация результатов
print("\n📈 10. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("-" * 40)

# Отдельные графики для линейного и RBF ядер
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# График для линейного ядра
linear_results = results_df[results_df['param_svc__kernel'] == 'linear']
ax1 = axes[0]
ax1.plot(param_range, linear_results['mean_test_score'], 
         marker='o', markersize=8, linewidth=2, color='blue', label='Средняя точность')
ax1.fill_between(param_range,
                 linear_results['mean_test_score'] + linear_results['std_test_score'],
                 linear_results['mean_test_score'] - linear_results['std_test_score'],
                 alpha=0.2, color='blue')
ax1.set_xscale('log')
ax1.set_xlabel('Параметр C (логарифмическая шкала)')
ax1.set_ylabel('Точность')
ax1.set_title('Линейное ядро SVM')
ax1.grid(True, alpha=0.3)
ax1.legend()

# График для RBF ядра (heatmap)
rbf_results = results_df[results_df['param_svc__kernel'] == 'rbf']
ax2 = axes[1]
# Создание матрицы для heatmap
rbf_pivot = rbf_results.pivot_table(
    index='param_svc__gamma',
    columns='param_svc__C',
    values='mean_test_score'
)
# Сортировка индексов и колонок для правильного порядка
rbf_pivot = rbf_pivot.sort_index(ascending=False).sort_index(axis=1, ascending=False)

im = ax2.imshow(rbf_pivot.values, cmap='YlOrRd', aspect='auto')
ax2.set_xticks(range(len(rbf_pivot.columns)))
ax2.set_yticks(range(len(rbf_pivot.index)))
ax2.set_xticklabels([f'{c:.1e}' for c in rbf_pivot.columns], rotation=45)
ax2.set_yticklabels([f'{g:.1e}' for g in rbf_pivot.index])
ax2.set_xlabel('Параметр C')
ax2.set_ylabel('Параметр gamma')
ax2.set_title('RBF ядро SVM (heatmap точности)')
plt.colorbar(im, ax=ax2, label='Точность')

plt.tight_layout()
plt.show()

# 11. Выводы
print("\n📝 11. ВЫВОДЫ")
print("=" * 60)
print("Поиск по сетке (GridSearchCV) позволяет:")
print("  ✅ Систематически перебирать комбинации гиперпараметров")
print("  ✅ Находить оптимальные значения для модели")
print("  ✅ Использовать перекрёстную проверку для надёжной оценки")
print("  ✅ Параллельно вычислять на нескольких ядрах (n_jobs=-1)")
print("  ✅ Автоматически обучать лучшую модель (refit=True)")
print("\nПараметры SVM:")
print("  - C: обратный параметр регуляризации")
print("    * Малые значения → сильная регуляризация → простая модель")
print("    * Большие значения → слабая регуляризация → сложная модель")
print("  - gamma: коэффициент ядра RBF")
print("    * Малые значения → широкое влияние → простая модель")
print("    * Большие значения → узкое влияние → сложная модель")
print("  - kernel: тип ядра")
print("    * linear: линейное разделение")
print("    * rbf: нелинейное разделение (радиальная базисная функция)")
print("\nПрактические рекомендации:")
print("  - Использовать логарифмическую шкалу для C и gamma")
print("  - Начинать с широкого диапазона значений")
print("  - После нахождения области оптимума сузить диапазон")
print("  - Учитывать вычислительные затраты при большом количестве комбинаций")
