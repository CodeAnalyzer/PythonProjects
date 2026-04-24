# -*- coding: utf-8 -*-
"""
Раздел 6.4.3: Поиск гиперпараметров методом последовательного деления пополам

Метод последовательного деления пополам (Successive Halving) - это ресурсоэффективный
алгоритм поиска гиперпараметров, который последовательно отбрасывает менее
перспективные конфигурации до тех пор, пока не останется только лучшая.

Алгоритм:
1. Формируем большой набор конфигураций-кандидатов с помощью случайной выборки
2. Обучаем модели с ограниченными ресурсами (небольшое подмножество данных)
3. Отбрасываем нижние X% моделей на основе их производительности
4. Повторяем с увеличенным количеством ресурсов до одной конфигурации

В современных версиях scikit-learn (1.2+) классы HalvingRandomSearchCV и
HalvingGridSearchCV уже стабилизированы и находятся в sklearn.model_selection,
без необходимости включения экспериментальных функций.
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
from sklearn.experimental import enable_halving_search_cv  # Включение экспериментального класса
from sklearn.model_selection import HalvingRandomSearchCV, train_test_split
from sklearn.pipeline import make_pipeline

print("⚡ ПОСЛЕДОВАТЕЛЬНОЕ ДЕЛЕНИЕ ПОПОЛАМ: HALVINGRANDOMSEARCHCV")
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

print("Сетка распределений гиперпараметров:")
print("  Конфигурация 1 (линейное ядро):")
print("    - svc__C: loguniform(0.0001, 1000.0)")
print("    - svc__kernel: ['linear']")
print("  Конфигурация 2 (RBF ядро):")
print("    - svc__C: loguniform(0.0001, 1000.0)")
print("    - svc__gamma: loguniform(0.0001, 1000.0)")
print("    - svc__kernel: ['rbf']")

# 5. Последовательное деление пополам
print("\n⚡ 5. ПОСЛЕДОВАТЕЛЬНОЕ ДЕЛЕНИЕ ПОПОЛАМ (HalvingRandomSearchCV)")
print("-" * 40)

hs = HalvingRandomSearchCV(
    estimator=pipe_svc,
    param_distributions=param_distributions,
    n_candidates='exhaust',
    resource='n_samples',
    factor=1.5,
    cv=10,
    scoring='accuracy',
    random_state=1,
    refit=True,
    n_jobs=-1
)

print("Запуск HalvingRandomSearchCV...")
print("  - n_candidates='exhaust' (автоматический выбор количества конфигураций)")
print("  - resource='n_samples' (ресурс - размер обучающего набора)")
print("  - factor=1.5 (каждый раунд проходит 100%/1.5 ≈ 66% кандидатов)")
print("  - cv=10 (10-кратная перекрёстная проверка)")
print("  - scoring='accuracy' (метрика точности)")
print("  - refit=True (автоматическое обучение лучшей модели)")
print("  - random_state=1 (воспроизводимость)")
print("  - n_jobs=-1 (использование всех ядер CPU)")

hs.fit(X_train, y_train)

print("\n✅ Поиск завершён!")

# 6. Результаты поиска
print("\n📈 6. РЕЗУЛЬТАТЫ ПОИСКА")
print("-" * 40)

print(f'Лучшая точность (CV): {hs.best_score_:.4f}')
print(f'\nЛучшие параметры:')
for param, value in hs.best_params_.items():
    print(f'  {param}: {value}')

# 7. Анализ раундов
print("\n📊 7. АНАЛИЗ РАУНДОВ ОТБОРА")
print("-" * 40)

results_df = pd.DataFrame(hs.cv_results_)
print(f'Всего раундов: {hs.n_resources_}')
print(f'Всего проверено конфигураций: {len(results_df)}')

# Информация по раундам
n_rounds = len(hs.n_resources_)
for i in range(n_rounds):
    round_results = results_df[results_df['iter'] == i]
    n_candidates = len(round_results)
    n_resources = hs.n_resources_[i]
    print(f'\nРаунд {i + 1}:')
    print(f'  Количество кандидатов: {n_candidates}')
    print(f'  Ресурс (образцов): {n_resources}')
    if i > 0:
        prev_candidates = len(results_df[results_df['iter'] == i - 1])
        kept_ratio = n_candidates / prev_candidates * 100
        print(f'  Прошло из предыдущего раунда: {kept_ratio:.1f}%')

# 8. Топ конфигураций по раундам
print("\n📊 8. ТОП КОНФИГУРАЦИЙ ПО РАУНДАМ")
print("-" * 40)

for i in range(n_rounds):
    round_results = results_df[results_df['iter'] == i]
    if len(round_results) > 0:
        best_in_round = round_results.nlargest(3, 'mean_test_score')[['params', 'mean_test_score', 'std_test_score']]
        print(f'\nРаунд {i + 1} - топ-3 конфигурации:')
        for idx, row in best_in_round.iterrows():
            params_str = str(row['params'])
            print(f"  {row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}: {params_str}")

# 9. Визуализация прогресса по раундам
print("\n📈 9. ВИЗУАЛИЗАЦИЯ ПРОГРЕССА ПО РАУНДАМ")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# График 1: Количество кандидатов по раундам
ax1 = axes[0, 0]
rounds = range(1, n_rounds + 1)
candidates_per_round = [len(results_df[results_df['iter'] == i]) for i in range(n_rounds)]
ax1.bar(rounds, candidates_per_round, color='steelblue', alpha=0.7, edgecolor='black')
ax1.set_xlabel('Номер раунда')
ax1.set_ylabel('Количество кандидатов')
ax1.set_title('Количество кандидатов по раундам')
ax1.grid(True, alpha=0.3, axis='y')

# График 2: Ресурс (образцов) по раундам
ax2 = axes[0, 1]
resources_per_round = hs.n_resources_
ax2.plot(rounds, resources_per_round, marker='o', linewidth=2, color='darkorange', markersize=8)
ax2.fill_between(rounds, 0, resources_per_round, alpha=0.3, color='darkorange')
ax2.set_xlabel('Номер раунда')
ax2.set_ylabel('Ресурс (количество образцов)')
ax2.set_title('Ресурс по раундам')
ax2.grid(True, alpha=0.3)

# График 3: Лучшая точность по раундам
ax3 = axes[1, 0]
best_score_per_round = []
for i in range(n_rounds):
    round_results = results_df[results_df['iter'] == i]
    best_score_per_round.append(round_results['mean_test_score'].max())
ax3.plot(rounds, best_score_per_round, marker='s', linewidth=2, color='green', markersize=8)
ax3.set_xlabel('Номер раунда')
ax3.set_ylabel('Лучшая точность (CV)')
ax3.set_title('Лучшая точность по раундам')
ax3.grid(True, alpha=0.3)
ax3.set_ylim([0.9, 1.0])

# График 4: Распределение точности в последнем раунде
ax4 = axes[1, 1]
final_round_results = results_df[results_df['iter'] == n_rounds - 1]
ax4.hist(final_round_results['mean_test_score'], bins=5, edgecolor='black', alpha=0.7, color='purple')
ax4.axvline(x=hs.best_score_, color='red', linestyle='--', linewidth=2, 
            label=f'Лучшая: {hs.best_score_:.4f}')
ax4.set_xlabel('Точность (CV)')
ax4.set_ylabel('Количество конфигураций')
ax4.set_title(f'Распределение точности в финальном раунде ({len(final_round_results)} конфигураций)')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# 10. Сравнение методов
print("\n📊 10. СРАВНЕНИЕ МЕТОДОВ ПОИСКА")
print("-" * 40)

print("Сравнение подходов к поиску гиперпараметров:")
print("\n  GridSearchCV:")
print("    - Перебирает ВСЕ комбинации из сетки")
print("    - Использует полные ресурсы для всех конфигураций")
print("    - Гарантирует нахождение оптимума в сетке")
print("    - Вычислительно затратно")
print("\n  RandomizedSearchCV:")
print("    - Случайный выбор конфигураций из распределений")
print("    - Использует полные ресурсы для всех конфигураций")
print("    - Эффективен при большом пространстве поиска")
print("    - Не гарантирует глобальный оптимум")
print("\n  HalvingRandomSearchCV:")
print("    - Последовательное деление кандидатов")
print("    - Начинает с ограниченных ресурсов")
print("    - Отбрасывает худшие конфигурации")
print("    - Увеличивает ресурсы для лучших кандидатов")
print("    - Наиболее ресурсоэффективный метод")

print(f"\nВ данном примере HalvingRandomSearchCV:")
print(f"  - Раундов: {n_rounds}")
print(f"  - Всего конфигураций: {len(results_df)}")
print(f"  - Финальный раунд: {len(results_df[results_df['iter'] == n_rounds - 1])} конфигураций")

# 11. Оценка на тестовом наборе
print("\n🎯 11. ФИНАЛЬНАЯ ОЦЕНКА НА ТЕСТОВОМ НАБОРЕ")
print("-" * 40)

# Лучшая модель уже обучена на полном обучающем наборе (refit=True)
clf = hs.best_estimator_
test_score = clf.score(X_test, y_test)

print(f'Точность на тестовом наборе: {test_score:.4f}')
print(f'Ошибок: {(clf.predict(X_test) != y_test).sum()} из {len(y_test)}')

# 12. Сравнение с HalvingGridSearchCV
print("\n🔬 12. СРАВНЕНИЕ С HALVINGGRIDSEARCHCV")
print("-" * 40)

print("HalvingGridSearchCV - альтернатива HalvingRandomSearchCV:")
print("  - Вместо случайной выборки использует ВСЕ конфигурации из сетки")
print("  - Полезен, когда пространство поиска небольшое")
print("  - Гарантирует рассмотрение всех комбинаций")
print("  - Более вычислительно затратно, чем HalvingRandomSearchCV")

# Пример с дискретными значениями для демонстрации
param_grid = [
    {
        'svc__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'svc__kernel': ['linear']
    },
    {
        'svc__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'svc__gamma': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'svc__kernel': ['rbf']
    }
]

print(f"\nПример сетки для HalvingGridSearchCV:")
print(f"  - Линейное ядро: 6 значений C")
print(f"  - RBF ядро: 6×6 = 36 комбинаций C и gamma")
print(f"  - Всего: 42 конфигурации")

print(f"\nПримечание:")
print(f"  В современных версиях scikit-learn (1.2+) классы")
print(f"  HalvingRandomSearchCV и HalvingGridSearchCV уже стабилизированы")
print(f"  и находятся в sklearn.model_selection без необходимости")
print(f"  включения экспериментальных функций.")

# 13. Выводы
print("\n📝 13. ВЫВОДЫ")
print("=" * 60)
print("Последовательное деление пополам (Successive Halving) позволяет:")
print("  ✅ Эффективно использовать вычислительные ресурсы")
print("  ✅ Быстро отбрасывать неперспективные конфигурации")
print("  ✅ Фокусировать ресурсы на лучших кандидатах")
print("  ✅ Находить хорошие конфигурации за меньшее время")
print("  ✅ Автоматически балансировать количество конфигураций и ресурсов")
print("\nАлгоритм работы:")
print("  1. Генерация большого набора конфигураций-кандидатов")
print("  2. Обучение на ограниченных ресурсах (малый набор данных)")
print("  3. Отбрасывание худших X% конфигураций")
print("  4. Повторение с увеличенными ресурсами")
print("  5. Продолжение до одной финальной конфигурации")
print("\nПараметры HalvingRandomSearchCV:")
print("  - factor: коэффициент отбора (factor=2 → 50% проходят)")
print("  - resource: тип ресурса ('n_samples' по умолчанию)")
print("  - n_candidates: количество конфигураций ('exhaust' - авто)")
print("  - min_resources: минимальное количество ресурсов")
print("\nКогда использовать HalvingRandomSearchCV:")
print("  - При очень большом пространстве гиперпараметров")
print("  - Когда вычислительные ресурсы сильно ограничены")
print("  - Когда нужна максимальная эффективность")
print("  - При непрерывных гиперпараметрах с широким диапазоном")
