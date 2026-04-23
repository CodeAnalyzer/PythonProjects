# -*- coding: utf-8 -*-
"""
Раздел 6.3.1: Диагностика смещения и дисперсии с помощью кривых обучения

Кривые обучения позволяют определить, страдает ли модель от высокой дисперсии
(переобучение) или высокого смещения (недообучение) и может ли сбор
дополнительных данных помочь решить эту проблему.

Если модель слишком сложна для имеющегося обучающего набора данных, она
обычно переобучается. Добавление обучающих данных может уменьшить риск
переобучения, но на практике найти дополнительные данные часто дорого или
невозможно.

Построив график точности модели на обучающем и проверочном наборах как
функцию от размера обучающего набора, мы можем диагностировать проблему:
- Высокое смещение: и обучающая, и валидационная точность низкие
- Высокая дисперсия: обучающая точность высокая, валидационная низкая
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
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.pipeline import make_pipeline

print("📊 КРИВЫЕ ОБУЧЕНИЯ: ДИАГНОСТИКА СМЕЩЕНИЯ И ДИСПЕРСИИ")
print("=" * 60)

# 1. Загрузка набора данных WDBC (Wisconsin Diagnostic Breast Cancer)
print("\n📂 1. ЗАГРУЗКА ДАННЫХ WDBC")
print("-" * 40)

df_wdbc = pd.read_csv('wdbc.data', header=None)

# Первая колонка - ID, вторая - диагноз (M/B), остальные - признаки
df_wdbc.columns = ['ID', 'Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]

# Кодирование диагноза: M (malignant) -> 1, B (benign) -> 0
le = LabelEncoder()
df_wdbc['Diagnosis'] = le.fit_transform(df_wdbc['Diagnosis'])

# Удаление колонки ID (не используется для обучения)
X = df_wdbc.iloc[:, 2:].values
y = df_wdbc['Diagnosis'].values

print(f'Классы: {np.unique(y)}')
print(f'Классовые метки: {le.classes_}')
print(f'Форма X: {X.shape}')
print(f'Форма y: {y.shape}')
print(f'\nРаспределение классов:')
for cls, label in zip(np.unique(y), le.classes_):
    count = np.sum(y == cls)
    print(f'  Класс {cls} ({label}): {count} образцов ({count/len(y)*100:.1f}%)')

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
# max_iter=10000 для избежания проблем сходимости на малых наборах данных
pipe_lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        C=1.0,
        solver='lbfgs',
        max_iter=10000,
        random_state=1
    )
)

print("Конвейер включает:")
print("  1. StandardScaler - стандартизация признаков")
print("  2. LogisticRegression - логистическая регрессия")
print("     - L2-регуляризация (по умолчанию)")
print("     - max_iter=10000 (для сходимости на малых наборах)")

# 4. Построение кривых обучения
print("\n📈 4. ПОСТРОЕНИЕ КРИВЫХ ОБУЧЕНИЯ")
print("-" * 40)

train_sizes, train_scores, test_scores = learning_curve(
    estimator=pipe_lr,
    X=X_train,
    y=y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=10,
    n_jobs=1,
    random_state=1
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

print(f'Размеры обучающих наборов: {train_sizes}')
print(f'\nСредняя точность на обучении:')
for size, score in zip(train_sizes, train_mean):
    print(f'  {size:3.0f} образцов: {score:.3f} ± {train_std[np.where(train_sizes == size)[0][0]]:.3f}')

print(f'\nСредняя точность на валидации:')
for size, score in zip(train_sizes, test_mean):
    print(f'  {size:3.0f} образцов: {score:.3f} ± {test_std[np.where(train_sizes == size)[0][0]]:.3f}')

# 5. Визуализация кривых обучения
print("\n📊 5. ВИЗУАЛИЗАЦИЯ КРИВЫХ ОБУЧЕНИЯ")
print("-" * 40)

plt.figure(figsize=(10, 6))

plt.plot(train_sizes, train_mean,
         color='blue', marker='o',
         markersize=5, label='Training accuracy')

plt.fill_between(train_sizes,
                 train_mean + train_std,
                 train_mean - train_std,
                 alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean,
         color='green', linestyle='--',
         marker='s', markersize=5,
         label='Validation accuracy')

plt.fill_between(train_sizes,
                 test_mean + test_std,
                 test_mean - test_std,
                 alpha=0.15, color='green')

plt.xlabel('Количество обучающих примеров')
plt.ylabel('Точность')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.03])
plt.title('Кривые обучения для логистической регрессии')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 6. Диагностика смещения и дисперсии
print("\n🔍 6. ДИАГНОСТИКА СМЕЩЕНИЯ И ДИСПЕРСИИ")
print("-" * 40)

# Анализ разрыва между обучающей и валидационной точностью
gap = train_mean[-1] - test_mean[-1]
print(f'Разрыв между обучающей и валидационной точностью (полный набор): {gap:.3f}')

if train_mean[-1] > 0.95 and test_mean[-1] < 0.85:
    print("\n⚠️  ВЫСОКАЯ ДИСПЕРСИЯ (ПЕРЕОБУЧЕНИЕ):")
    print("   - Обучающая точность значительно выше валидационной")
    print("   - Модель слишком сложна для имеющихся данных")
    print("   - Рекомендации:")
    print("     * Собрать больше обучающих данных")
    print("     * Уменьшить сложность модели (регуляризация, меньше признаков)")
    print("     * Использовать более простую модель")
elif train_mean[-1] < 0.85 and test_mean[-1] < 0.85:
    print("\n⚠️  ВЫСОКОЕ СМЕЩЕНИЕ (НЕДОБУЧЕНИЕ):")
    print("   - И обучающая, и валидационная точность низкие")
    print("   - Модель слишком проста")
    print("   - Рекомендации:")
    print("     * Увеличить сложность модели")
    print("     * Добавить новые признаки")
    print("     * Уменьшить регуляризацию")
elif gap < 0.05:
    print("\n✅ ХОРОШИЙ БАЛАНС:")
    print("   - Обучающая и валидационная точность близки")
    print("   - Модель хорошо обобщает")
    print("   - Рекомендации:")
    print("     * Можно попробовать собрать больше данных для улучшения")
    print("     * Или модель уже достаточно хороша")
else:
    print("\n⚠️  УМЕРЕННАЯ ДИСПЕРСИЯ:")
    print("   - Наблюдается небольшой разрыв между точностями")
    print("   - Рекомендации:")
    print("     * Попробовать собрать больше данных")
    print("     * Или немного уменьшить сложность модели")

# 7. Анализ динамики при увеличении данных
print("\n📊 7. АНАЛИЗ ДИНАМИКИ ПРИ УВЕЛИЧЕНИИ ДАННЫХ")
print("-" * 40)

# Улучшение валидационной точности от минимального к максимальному размеру
improvement = test_mean[-1] - test_mean[0]
print(f'Улучшение валидационной точности: {improvement:.3f}')

if improvement > 0.05:
    print("✅ Сбор дополнительных данных может значительно улучшить модель")
elif improvement > 0.02:
    print("⚠️  Сбор дополнительных данных может умеренно улучшить модель")
else:
    print("⚠️  Сбор дополнительных данных вряд ли значительно улучшит модель")

# 8. Финальная оценка на тестовом наборе
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
print("Кривые обучения позволяют:")
print("  ✅ Диагностировать высокое смещение (недообучение)")
print("  ✅ Диагностировать высокую дисперсию (переобучение)")
print("  ✅ Оценить потенциальную пользу от сбора дополнительных данных")
print("  ✅ Определить оптимальный размер обучающего набора")
print("\nИнтерпретация кривых:")
print("  - Если обе кривые низкие → высокое смещение → усложнить модель")
print("  - Если обучающая высокая, валидационная низкая → высокая дисперсия")
print("    → собрать больше данных или упростить модель")
print("  - Если обе кривые близки и высокие → хороший баланс")
print("\nПараметр max_iter=10000:")
print("  - Необходим для сходимости на малых обучающих наборах")
print("  - По умолчанию LogisticRegression использует 1000 итераций")
