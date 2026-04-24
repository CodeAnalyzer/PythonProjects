# -*- coding: utf-8 -*-
"""
Раздел 6.4.2: Изучение обширных конфигураций гиперпараметров с помощью рандомизированного поиска

Рандомизированный поиск (Randomized Search) - альтернатива поиску по сетке, которая
случайным образом выбирает конфигурации гиперпараметров из распределений.

Преимущества перед GridSearchCV:
- Более эффективен с точки зрения вычислительных затрат
- Позволяет исследовать более широкий диапазон значений
- Может найти хорошие конфигурации, которые GridSearch мог пропустить
- Использует распределения вероятностей вместо дискретных списков

Логарифмическое распределение (loguniform) гарантирует, что из каждого
логарифмического диапазона будет взято примерно одинаковое количество выборок.
"""
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import make_pipeline

print("🎲 РАНДОМИЗИРОВАННЫЙ ПОИСК: НАСТРОЙКА ГИПЕРПАРАМЕТРОВ SVM")
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

# 4. Определение распределений гиперпараметров
print("\n📊 4. ОПРЕДЕЛЕНИЕ РАСПРЕДЕЛЕНИЙ ГИПЕРПАРАМЕТРОВ")
print("-" * 40)

# Логарифмическое распределение от 0.0001 до 1000.0
param_range = scipy.stats.loguniform(0.0001, 1000.0)

print("Использование логарифмического распределения (loguniform):")
print(f"  Диапазон: [{0.0001}, {1000.0}]")
print("\nЛогарифмическое распределение гарантирует, что из каждого")
print("логарифмического диапазона будет взято одинаковое количество выборок.")
print("\nПример 10 случайных выборок из распределения:")
np.random.seed(1)
samples = param_range.rvs(10)
for i, sample in enumerate(samples, 1):
    print(f"  Выборка {i}: {sample:.6e}")

param_distributions = [
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

print("\nСетка распределений гиперпараметров:")
print("  Конфигурация 1 (линейное ядро):")
print("    - svc__C: loguniform(0.0001, 1000.0)")
print("    - svc__kernel: ['linear']")
print("  Конфигурация 2 (RBF ядро):")
print("    - svc__C: loguniform(0.0001, 1000.0)")
print("    - svc__gamma: loguniform(0.0001, 1000.0)")
print("    - svc__kernel: ['rbf']")

# 5. Рандомизированный поиск
print("\n🎲 5. РАНДОМИЗИРОВАННЫЙ ПОИСК (RandomizedSearchCV)")
print("-" * 40)

n_iter = 20
rs = RandomizedSearchCV(
    estimator=pipe_svc,
    param_distributions=param_distributions,
    scoring='accuracy',
    cv=10,
    refit=True,
    n_iter=n_iter,
    random_state=1,
    n_jobs=-1
)

print(f"Запуск рандомизированного поиска...")
print(f"  - n_iter={n_iter} (количество случайных конфигураций)")
print(f"  - cv=10 (10-кратная перекрёстная проверка)")
print(f"  - scoring='accuracy' (метрика точности)")
print(f"  - refit=True (автоматическое обучение лучшей модели)")
print(f"  - random_state=1 (воспроизводимость)")
print(f"  - n_jobs=-1 (использование всех ядер CPU)")

rs.fit(X_train, y_train)

print("\n✅ Поиск завершён!")

# 6. Результаты поиска
print("\n📈 6. РЕЗУЛЬТАТЫ ПОИСКА")
print("-" * 40)

print(f'Лучшая точность (CV): {rs.best_score_:.4f}')
print(f'\nЛучшие параметры:')
for param, value in rs.best_params_.items():
    print(f'  {param}: {value}')

# 7. Анализ всех результатов
print("\n📊 7. АНАЛИЗ ВСЕХ РЕЗУЛЬТАТОВ")
print("-" * 40)

results_df = pd.DataFrame(rs.cv_results_)
print(f'Всего проверено конфигураций: {len(results_df)}')

# Топ-5 лучших конфигураций
print('\nТоп-5 лучших конфигураций:')
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

# 8. Визуализация распределения протестированных значений
print("\n📈 8. ВИЗУАЛИЗАЦИЯ РАСПРЕДЕЛЕНИЯ ЗНАЧЕНИЙ")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# График 1: Распределение значений C (все)
ax1 = axes[0, 0]
c_values = results_df['param_svc__C'].dropna()
ax1.hist(np.log10(c_values), bins=10, edgecolor='black', alpha=0.7, color='blue')
ax1.set_xlabel('log10(C)')
ax1.set_ylabel('Количество')
ax1.set_title('Распределение протестированных значений C (логарифмическая шкала)')
ax1.grid(True, alpha=0.3)

# График 2: Распределение значений gamma (только RBF)
ax2 = axes[0, 1]
gamma_values = results_df['param_svc__gamma'].dropna()
if len(gamma_values) > 0:
    ax2.hist(np.log10(gamma_values), bins=10, edgecolor='black', alpha=0.7, color='green')
    ax2.set_xlabel('log10(gamma)')
    ax2.set_ylabel('Количество')
    ax2.set_title('Распределение протестированных значений gamma (логарифмическая шкала)')
    ax2.grid(True, alpha=0.3)

# График 3: Точность vs log10(C)
ax3 = axes[1, 0]
scatter = ax3.scatter(np.log10(c_values), results_df['mean_test_score'][:len(c_values)], 
                      c=results_df['param_svc__kernel'][:len(c_values)].map({'linear': 'blue', 'rbf': 'red'}),
                      alpha=0.7, s=100)
ax3.set_xlabel('log10(C)')
ax3.set_ylabel('Точность (CV)')
ax3.set_title('Точность vs log10(C)')
ax3.grid(True, alpha=0.3)
# Добавляем легенду
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='blue', label='Linear'),
                   Patch(facecolor='red', label='RBF')]
ax3.legend(handles=legend_elements)

# График 4: Точность по итерациям
ax4 = axes[1, 1]
iterations = range(1, len(results_df) + 1)
ax4.plot(iterations, results_df['mean_test_score'], marker='o', linestyle='-', 
         color='purple', alpha=0.7)
ax4.axhline(y=rs.best_score_, color='red', linestyle='--', 
            label=f'Лучшая точность: {rs.best_score_:.4f}')
ax4.set_xlabel('Номер итерации')
ax4.set_ylabel('Точность (CV)')
ax4.set_title('Точность по итерациям рандомизированного поиска')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 9. Сравнение с GridSearchCV
print("\n📊 9. СРАВНЕНИЕ С GRIDSEARCHCV")
print("-" * 40)

print("Сравнение подходов:")
print("  GridSearchCV:")
print("    - Перебирает ВСЕ комбинации из сетки")
print("    - Гарантирует нахождение оптимальной конфигурации в сетке")
print("    - Вычислительно затратно при большом количестве комбинаций")
print("\n  RandomizedSearchCV:")
print("    - Выбирает СЛУЧАЙНЫЕ конфигурации из распределений")
print("    - Может найти хорошие конфигурации за меньшее время")
print("    - Позволяет исследовать более широкий диапазон значений")
print("    - Не гарантирует нахождение глобального оптимума")

print(f"\nВ данном примере:")
print(f"  - RandomizedSearchCV проверил {n_iter} конфигураций")
print(f"  - При GridSearchCV с теми же диапазонами было бы проверено")
print(f"    8 (C) + 8*8 (C*gamma) = 72 конфигурации")
print(f"  - Экономия вычислений: {(72 - n_iter) / 72 * 100:.1f}%")

# 10. Оценка на тестовом наборе
print("\n🎯 10. ФИНАЛЬНАЯ ОЦЕНКА НА ТЕСТОВОМ НАБОРЕ")
print("-" * 40)

# Лучшая модель уже обучена на полном обучающем наборе (refit=True)
clf = rs.best_estimator_
test_score = clf.score(X_test, y_test)

print(f'Точность на тестовом наборе: {test_score:.4f}')
print(f'Ошибок: {(clf.predict(X_test) != y_test).sum()} из {len(y_test)}')

# 11. Дополнительный эксперимент: больше итераций
print("\n🔬 11. ДОПОЛНИТЕЛЬНЫЙ ЭКСПЕРИМЕНТ: БОЛЬШЕ ИТЕРАЦИЙ")
print("-" * 40)

print("Попробуем увеличить количество итераций для лучшего поиска:")
n_iter_large = 50
rs_large = RandomizedSearchCV(
    estimator=pipe_svc,
    param_distributions=param_distributions,
    scoring='accuracy',
    cv=10,
    refit=True,
    n_iter=n_iter_large,
    random_state=1,
    n_jobs=-1
)

print(f"Запуск поиска с n_iter={n_iter_large}...")
rs_large.fit(X_train, y_train)

print(f"\nРезультаты с {n_iter_large} итерациями:")
print(f'  Лучшая точность (CV): {rs_large.best_score_:.4f}')
print(f'  Лучшие параметры: {rs_large.best_params_}')

clf_large = rs_large.best_estimator_
test_score_large = clf_large.score(X_test, y_test)

print(f'  Точность на тестовом наборе: {test_score_large:.4f}')

improvement = (rs_large.best_score_ - rs.best_score_) * 100
if improvement > 0:
    print(f'\n✅ Увеличение итераций улучшило CV точность на {improvement:.2f}%')
else:
    print(f'\n⚠️  Увеличение итераций не улучшило CV точность')

# 12. Выводы
print("\n📝 12. ВЫВОДЫ")
print("=" * 60)
print("Рандомизированный поиск (RandomizedSearchCV) позволяет:")
print("  ✅ Исследовать широкий диапазон гиперпараметров")
print("  ✅ Использовать распределения вероятностей вместо дискретных списков")
print("  ✅ Экономить вычислительные ресурсы по сравнению с GridSearchCV")
print("  ✅ Находить хорошие конфигурации за меньшее время")
print("  ✅ Эффективно работать с непрерывными гиперпараметрами")
print("\nЛогарифмическое распределение (loguniform):")
print("  - Гарантирует равномерное покрытие логарифмических диапазонов")
print("  - Идеально подходит для параметров регуляризации (C, gamma)")
print("  - Из диапазона [0.0001, 0.001] будет взято столько же выборок,")
print("    сколько и из [10.0, 100.0]")
print("\nКогда использовать RandomizedSearchCV:")
print("  - При большом пространстве гиперпараметров")
print("  - Когда вычислительные ресурсы ограничены")
print("  - Когда нужно быстро получить хорошую конфигурацию")
print("  - При непрерывных гиперпараметрах")
print("\nКогда использовать GridSearchCV:")
print("  - При небольшом пространстве гиперпараметров")
print("  - Когда нужно гарантированно найти оптимум в заданной сетке")
print("  - При категориальных гиперпараметрах")
print("  - Когда вычислительные ресурсы не ограничены")
