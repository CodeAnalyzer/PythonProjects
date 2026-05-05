# -*- coding: utf-8 -*-
"""
Раздел 6.5.1: Чтение матрицы несоответствий

Матрица несоответствий (Confusion Matrix) - это квадратная матрица, которая
показывает производительность классификатора путём сравнения фактических
и предсказанных меток классов.

Матрица содержит 4 значения:
- True Positive (TP): правильно предсказанные положительные примеры
- True Negative (TN): правильно предсказанные отрицательные примеры
- False Positive (FP): отрицательные примеры, ошибочно предсказанные как положительные
- False Negative (FN): положительные примеры, ошибочно предсказанные как отрицательные

Для бинарной классификации матрица имеет вид:
                Предсказано
               0      1
Истинно  0  [[TN,   FP]
          1   [FN,   TP]]
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
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

print("📊 МАТРИЦА НЕСООТВЕТСТВИЙ (CONFUSION MATRIX)")
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
print(f'  Класс 0: {le.classes_[0]} (доброкачественная)')
print(f'  Класс 1: {le.classes_[1]} (злокачественная)')
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

# 3. Создание и обучение модели SVM
print("\n🔧 3. ОБУЧЕНИЕ МОДЕЛИ SVM")
print("-" * 40)

pipe_svc = make_pipeline(
    StandardScaler(),
    SVC(random_state=1)
)

pipe_svc.fit(X_train, y_train)

print("Модель SVM обучена на обучающем наборе")

# 4. Предсказание на тестовом наборе
print("\n🎯 4. ПРЕДСКАЗАНИЕ НА ТЕСТОВОМ НАБОРЕ")
print("-" * 40)

y_pred = pipe_svc.predict(X_test)

print(f'Предсказания выполнены для {len(y_test)} образцов')

# 5. Вычисление матрицы несоответствий
print("\n📊 5. ВЫЧИСЛЕНИЕ МАТРИЦЫ НЕСООТВЕТСТВИЙ")
print("-" * 40)

confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)

print("Матрица несоответствий:")
print(confmat)

# Разбор значений матрицы
tn, fp, fn, tp = confmat.ravel()

print(f'\nРазбор значений:')
print(f'  True Negative (TN): {tn} - доброкачественные, предсказаны как доброкачественные')
print(f'  False Positive (FP): {fp} - доброкачественные, предсказаны как злокачественные')
print(f'  False Negative (FN): {fn} - злокачественные, предсказаны как доброкачественные')
print(f'  True Positive (TP): {tp} - злокачественные, предсказаны как злокачественные')

# 6. Визуализация матрицы несоответствий
print("\n📈 6. ВИЗУАЛИЗАЦИЯ МАТРИЦЫ НЕСООТВЕТСТВИЙ")
print("-" * 40)

fig, ax = plt.subplots(figsize=(5, 4))
ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)

# Добавление текстовых значений в ячейки
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j, y=i, s=confmat[i, j], 
                va='center', ha='center', 
                fontsize=14, fontweight='bold')

# Настройка осей
ax.xaxis.set_ticks_position('bottom')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Доброкачественная (0)', 'Злокачественная (1)'], rotation=45, ha='right')
ax.set_yticklabels(['Доброкачественная (0)', 'Злокачественная (1)'])
ax.set_xlabel('Предсказанная метка')
ax.set_ylabel('Истинная метка')
ax.set_title('Матрица несоответствий для SVM')

plt.tight_layout()
plt.show()

# 7. Дополнительная визуализация с аннотациями
print("\n📈 7. ДОПОЛНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ С АННОТАЦИЯМИ")
print("-" * 40)

fig, ax = plt.subplots(figsize=(6, 5))

# Создание heatmap
im = ax.imshow(confmat, cmap=plt.cm.Blues, alpha=0.7)

# Добавление цветовой шкалы
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Количество образцов', rotation=270, labelpad=20)

# Добавление текстовых значений с пояснениями
annotations = [
    ('TN\n(Правильно)', 'FP\n(Ошибка типа I)'),
    ('FN\n(Ошибка типа II)', 'TP\n(Правильно)')
]

for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        text = f'{confmat[i, j]}\n{annotations[i][j]}'
        ax.text(x=j, y=i, s=text, 
                va='center', ha='center', 
                fontsize=12, fontweight='bold',
                color='white' if confmat[i, j] > confmat.max() / 2 else 'black')

# Настройка осей
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(['Доброкачественная (0)', 'Злокачественная (1)'], rotation=45, ha='right')
ax.set_yticklabels(['Доброкачественная (0)', 'Злокачественная (1)'])
ax.set_xlabel('Предсказанная метка', fontsize=12, fontweight='bold')
ax.set_ylabel('Истинная метка', fontsize=12, fontweight='bold')
ax.set_title('Матрица несоответствий с пояснениями', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.show()

# 8. Вычисление метрик на основе матрицы несоответствий
print("\n📊 8. ВЫЧИСЛЕНИЕ МЕТРИК НА ОСНОВЕ МАТРИЦЫ")
print("-" * 40)

# Точность (Accuracy)
accuracy = (tp + tn) / (tp + tn + fp + fn)
print(f'Точность (Accuracy): {accuracy:.4f}')
print(f'  = (TP + TN) / (TP + TN + FP + FN)')
print(f'  = ({tp} + {tn}) / ({tp} + {tn} + {fp} + {fn})')

# Точность положительного класса (Precision)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
print(f'\nТочность (Precision): {precision:.4f}')
print(f'  = TP / (TP + FP)')
print(f'  = {tp} / ({tp} + {fp})')
print(f'  Из всех предсказанных как злокачественные, {precision*100:.1f}% действительно злокачественные')

# Полнота (Recall/Sensitivity)
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
print(f'\nПолнота (Recall/Sensitivity): {recall:.4f}')
print(f'  = TP / (TP + FN)')
print(f'  = {tp} / ({tp} + {fn})')
print(f'  Из всех реально злокачественных, {recall*100:.1f}% были найдены')

# Специфичность (Specificity)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
print(f'\nСпецифичность (Specificity): {specificity:.4f}')
print(f'  = TN / (TN + FP)')
print(f'  = {tn} / ({tn} + {fp})')
print(f'  Из всех реально доброкачественных, {specificity*100:.1f}% были правильно классифицированы')

# F1-score
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
print(f'\nF1-score: {f1_score:.4f}')
print(f'  = 2 * (Precision * Recall) / (Precision + Recall)')
print(f'  = 2 * ({precision:.4f} * {recall:.4f}) / ({precision:.4f} + {recall:.4f})')
print(f'  Гармоническое среднее precision и recall')

# 9. Интерпретация ошибок
print("\n🔍 9. ИНТЕРПРЕТАЦИЯ ОШИБОК")
print("-" * 40)

print("Анализ ошибок:")
print(f'\nЛожноположительные (FP): {fp}')
print("  - Доброкачественные опухоли, ошибочно предсказанные как злокачественные")
print("  - Это ошибка типа I (ложная тревога)")
print("  - В медицинской диагностике это может привести к лишним процедурам")
print("  - Но лучше пропустить FP, чем FN (лучше перестраховаться)")

print(f'\nЛожноотрицательные (FN): {fn}')
print("  - Злокачественные опухоли, ошибочно предсказанные как доброкачественные")
print("  - Это ошибка типа II (пропуск цели)")
print("  - В медицинской диагностике это критическая ошибка")
print("  - Может привести к пропуску лечения и ухудшению prognosis")

# 10. Сравнение с идеальной матрицей
print("\n📊 10. СРАВНЕНИЕ С ИДЕАЛЬНОЙ МАТРИЦЕЙ")
print("-" * 40)

# Идеальная матрица (без ошибок)
class_0_count = np.sum(y_test == 0)
class_1_count = np.sum(y_test == 1)
ideal_confmat = np.array([[class_0_count, 0], [0, class_1_count]])

print("Идеальная матрица (без ошибок):")
print(ideal_confmat)
print(f'\nТекущая матрица:')
print(confmat)

# Количество ошибок
total_errors = fp + fn
total_samples = len(y_test)
error_rate = total_errors / total_samples

print(f'\nОбщее количество ошибок: {total_errors} из {total_samples}')
print(f'Коэффициент ошибок: {error_rate:.4f} ({error_rate*100:.2f}%)')

# 11. Визуализация сравнения с идеальной матрицей
print("\n📈 11. ВИЗУАЛИЗАЦИЯ СРАВНЕНИЯ")
print("-" * 40)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Идеальная матрица
ax1 = axes[0]
ax1.matshow(ideal_confmat, cmap=plt.cm.Greens, alpha=0.3)
for i in range(ideal_confmat.shape[0]):
    for j in range(ideal_confmat.shape[1]):
        ax1.text(x=j, y=i, s=ideal_confmat[i, j], 
                va='center', ha='center', fontsize=14, fontweight='bold')
ax1.xaxis.set_ticks_position('bottom')
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['Доброкачественная', 'Злокачественная'], rotation=45, ha='right')
ax1.set_yticklabels(['Доброкачественная', 'Злокачественная'])
ax1.set_xlabel('Предсказанная метка')
ax1.set_ylabel('Истинная метка')
ax1.set_title('Идеальная матрица (без ошибок)')

# Текущая матрица
ax2 = axes[1]
ax2.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax2.text(x=j, y=i, s=confmat[i, j], 
                va='center', ha='center', fontsize=14, fontweight='bold')
ax2.xaxis.set_ticks_position('bottom')
ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Доброкачественная', 'Злокачественная'], rotation=45, ha='right')
ax2.set_yticklabels(['Доброкачественная', 'Злокачественная'])
ax2.set_xlabel('Предсказанная метка')
ax2.set_ylabel('Истинная метка')
ax2.set_title(f'Текущая матрица ({total_errors} ошибок)')

plt.tight_layout()
plt.show()

# 12. Выводы
print("\n📝 12. ВЫВОДЫ")
print("=" * 60)
print("Матрица несоответствий позволяет:")
print("  ✅ Визуализировать производительность классификатора")
print("  ✅ Вычислять различные метрики оценки")
print("  ✅ Понимать типы ошибок (FP и FN)")
print("  ✅ Оптимизировать модель под конкретные задачи")
print("\nКлючевые метрики:")
print("  - Accuracy: общая точность классификации")
print("  - Precision: точность предсказания положительного класса")
print("  - Recall: полнота обнаружения положительного класса")
print("  - Specificity: точность предсказания отрицательного класса")
print("  - F1-score: гармоническое среднее precision и recall")
print("\nТипы ошибок:")
print("  - False Positive (FP): ошибка типа I, ложная тревога")
print("  - False Negative (FN): ошибка типа II, пропуск цели")
print("\nВ медицинской диагностике:")
print("  - FN критичнее FP (лучше перестраховаться)")
print("  - Высокий recall важнее высокой precision")
print("  - F1-score балансирует precision и recall")
