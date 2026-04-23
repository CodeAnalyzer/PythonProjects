# -*- coding: utf-8 -*-
"""
Раздел 6.3.2: Устранение переобучения и недообучения с помощью кривых валидации

Кривые валидации - полезный инструмент для улучшения качества модели путем
решения проблем, вызванных как переобучением, так и недообучением.

В отличие от кривых обучения (которые зависят от размера набора данных),
кривые валидации варьируют значения параметров модели - например, параметр
регуляризации C в логистической регрессии.

Это позволяет найти оптимальное значение параметра, которое обеспечивает
наилучший баланс между смещением и дисперсией.
"""
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import validation_curve, train_test_split
from sklearn.pipeline import make_pipeline

print("📊 КРИВЫЕ ВАЛИДАЦИИ: ПОДБОР ПАРАМЕТРОВ РЕГУЛЯРИЗАЦИИ")
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

pipe_lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        solver='lbfgs',
        max_iter=10000,
        random_state=1
    )
)

print("Конвейер включает:")
print("  1. StandardScaler - стандартизация признаков")
print("  2. LogisticRegression - логистическая регрессия")
print("\nПараметр C (обратная регуляризация):")
print("  - Малые значения C (0.001, 0.01) → сильная регуляризация → простая модель")
print("  - Большие значения C (10, 100) → слабая регуляризация → сложная модель")
print("  - Оптимальные значения C → баланс между смещением и дисперсией")

# 4. Построение кривых валидации
print("\n📈 4. ПОСТРОЕНИЕ КРИВЫХ ВАЛИДАЦИИ")
print("-" * 40)

param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

train_scores, test_scores = validation_curve(
    estimator=pipe_lr,
    X=X_train,
    y=y_train,
    param_name='logisticregression__C',  # Двойное подчёркивание для доступа в pipeline
    param_range=param_range,
    cv=10
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

print(f'Диапазон параметров C: {param_range}')
print(f'\nСредняя точность на обучении:')
for c, score, std in zip(param_range, train_mean, train_std):
    print(f'  C={c:7.3f}: {score:.3f} ± {std:.3f}')

print(f'\nСредняя точность на валидации:')
for c, score, std in zip(param_range, test_mean, test_std):
    print(f'  C={c:7.3f}: {score:.3f} ± {std:.3f}')

# 5. Визуализация кривых валидации
print("\n📊 5. ВИЗУАЛИЗАЦИЯ КРИВЫХ ВАЛИДАЦИИ")
print("-" * 40)

plt.figure(figsize=(10, 6))

plt.plot(param_range, train_mean,
         color='blue', marker='o',
         markersize=5, label='Точность обучения')

plt.fill_between(param_range,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(param_range, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Точность валидации')

plt.fill_between(param_range,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Параметр C (логарифмическая шкала)')
plt.ylabel('Точность')
plt.ylim([0.8, 1.03])
plt.title('Кривые валидации для логистической регрессии')
plt.tight_layout()
plt.show()

# 6. Анализ результатов
print("\n🔍 6. АНАЛИЗ РЕЗУЛЬТАТОВ")
print("-" * 40)

# Поиск оптимального значения C
best_idx = np.argmax(test_mean)
best_c = param_range[best_idx]
best_score = test_mean[best_idx]

print(f'Оптимальное значение C: {best_c}')
print(f'Лучшая точность валидации: {best_score:.3f}')

# Анализ недообучения (малые значения C)
print(f'\nАнализ при C={param_range[0]} (сильная регуляризация):')
print(f'  Точность обучения: {train_mean[0]:.3f}')
print(f'  Точность валидации: {test_mean[0]:.3f}')
if train_mean[0] < 0.95 and test_mean[0] < 0.95:
    print('  ⚠️  Недообучение: модель слишком проста')

# Анализ переобучения (большие значения C)
print(f'\nАнализ при C={param_range[-1]} (слабая регуляризация):')
print(f'  Точность обучения: {train_mean[-1]:.3f}')
print(f'  Точность валидации: {test_mean[-1]:.3f}')
gap = train_mean[-1] - test_mean[-1]
if gap > 0.05:
    print(f'  ⚠️  Переобучение: разрыв {gap:.3f} между обучением и валидацией')
else:
    print('  ✅ Баланс: разрыв между обучением и валидацией мал')

# Определение области компромисса
print(f'\nОбласть компромисса (оптимальные значения C):')
compromise_range = []
for i, (train_score, test_score) in enumerate(zip(train_mean, test_mean)):
    gap = train_score - test_score
    if gap < 0.05 and test_score > 0.95:
        compromise_range.append(param_range[i])

if compromise_range:
    print(f'  Рекомендуемый диапазон: {compromise_range}')
else:
    # Находим диапазон с наименьшим разрывом и высокой точностью
    best_range_idx = np.argmin(np.abs(train_mean - test_mean))
    print(f'  Рекомендуемое значение: {param_range[best_range_idx]}')

# 7. Финальная оценка с оптимальным параметром
print("\n🎯 7. ФИНАЛЬНАЯ ОЦЕНКА С ОПТИМАЛЬНЫМ ПАРАМЕТРОМ")
print("-" * 40)

pipe_lr_opt = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        C=best_c,
        solver='lbfgs',
        max_iter=10000,
        random_state=1
    )
)

pipe_lr_opt.fit(X_train, y_train)
y_pred = pipe_lr_opt.predict(X_test)
test_accuracy = np.mean(y_pred == y_test)

print(f'Оптимальный параметр C: {best_c}')
print(f'Точность на тестовом наборе: {test_accuracy:.3f}')
print(f'Ошибок: {(y_pred != y_test).sum()} из {len(y_test)}')

# 8. Сравнение с дефолтным C=1.0
print("\n📊 8. СРАВНЕНИЕ С ДЕФОЛТНЫМ C=1.0")
print("-" * 40)

pipe_lr_default = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        C=1.0,
        solver='lbfgs',
        max_iter=10000,
        random_state=1
    )
)

pipe_lr_default.fit(X_train, y_train)
y_pred_default = pipe_lr_default.predict(X_test)
test_accuracy_default = np.mean(y_pred_default == y_test)

print(f'Параметр C: 1.0 (дефолт)')
print(f'Точность на тестовом наборе: {test_accuracy_default:.3f}')
print(f'Ошибок: {(y_pred_default != y_test).sum()} из {len(y_test)}')

if test_accuracy > test_accuracy_default:
    improvement = (test_accuracy - test_accuracy_default) * 100
    print(f'\n✅ Оптимизация улучшила точность на {improvement:.2f}%')
elif test_accuracy < test_accuracy_default:
    degradation = (test_accuracy_default - test_accuracy) * 100
    print(f'\n⚠️  Дефолтный параметр лучше на {degradation:.2f}%')
else:
    print(f'\n✅ Точность одинакова - оба параметра хороши')

# 9. Выводы
print("\n📝 9. ВЫВОДЫ")
print("=" * 60)
print("Кривые валидации позволяют:")
print("  ✅ Найти оптимальные значения гиперпараметров модели")
print("  ✅ Диагностировать переобучение (слабая регуляризация)")
print("  ✅ Диагностировать недообучение (сильная регуляризация)")
print("  ✅ Определить область компромисса между смещением и дисперсией")
print("\nИнтерпретация кривых:")
print("  - Малые значения C → сильная регуляризация → недообучение")
print("  - Большие значения C → слабая регуляризация → переобучение")
print("  - Оптимальные значения C → баланс между точностью и обобщением")
print("\nПрактические рекомендации:")
print("  - Использовать логарифмическую шкалу для параметров регуляризации")
print("  - Смотреть на разрыв между обучающей и валидационной точностью")
print("  - Выбирать значение с высокой валидационной точностью и малым разрывом")
