# -*- coding: utf-8 -*-
"""
Раздел 6.5.3: Построение рабочей характеристики приемника (ROC)

ROC-кривая (Receiver Operating Characteristic) - это график, показывающий
зависимость между долей ложноположительных (FPR) и долей истинно положительных (TPR)
при различных порогах принятия решения классификатора.

Интерпретация ROC-кривой:
- Диагональная линия (FPR = TPR): случайное угадывание (AUC = 0.5)
- Верхний левый угол (FPR=0, TPR=1): идеальный классификатор (AUC = 1.0)
- Ниже диагонали: хуже случайного угадывания (AUC < 0.5)
- Выше диагонали: лучше случайного угадывания (AUC > 0.5)

ROC AUC (Area Under the Curve) - площадь под ROC-кривой:
- 1.0: идеальный классификатор
- 0.5: случайное угадывание
- 0.0: полностью неверный классификатор
- Чем ближе к 1.0, тем лучше модель

Precision-Recall кривая - альтернатива ROC для несбалансированных классов:
- Показывает зависимость precision от recall при различных порогах
- Более информативна, когда положительный класс редкий
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
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline

print("📊 ПОСТРОЕНИЕ ROC-КРИВОЙ (RECEIVER OPERATING CHARACTERISTIC)")
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

# 3. Использование только двух признаков для усложнения задачи
print("\n🎯 3. ВЫБОР ДВУХ ПРИЗНАКОВ ДЛЯ УСЛОЖНЕНИЯ ЗАДАЧИ")
print("-" * 40)

# Выбираем признаки 4 и 14 (индексация с 0, поэтому 4 и 14)
X_train2 = X_train[:, [4, 14]]
X_test2 = X_test[:, [4, 14]]

print(f'Выбранные признаки: индексы [4, 14]')
print(f'  Признак 5 (индекс 4): {df_wdbc.columns[5]}')
print(f'  Признак 15 (индекс 14): {df_wdbc.columns[15]}')
print(f'\nФорма X_train2: {X_train2.shape}')
print(f'Форма X_test2: {X_test2.shape}')

# 4. Создание конвейера с LogisticRegression и PCA
print("\n🔧 4. СОЗДАНИЕ КОНВЕЙЕРА")
print("-" * 40)

pipe_lr = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    LogisticRegression(C=100.0, solver='lbfgs', random_state=1, max_iter=10000)
)

print("Конвейер включает:")
print("  1. StandardScaler - стандартизация признаков")
print("  2. PCA(n_components=2) - уменьшение размерности до 2")
print("  3. LogisticRegression - логистическая регрессия")

# 5. Построение ROC-кривой с перекрёстной проверкой
print("\n📈 5. ПОСТРОЕНИЕ ROC-КРИВОЙ С ПЕРЕКРЁСТНОЙ ПРОВЕРКОЙ")
print("-" * 40)

cv = StratifiedKFold(n_splits=3)
fig = plt.figure(figsize=(7, 5))

mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
all_tpr = []

print("Выполнение 3-кратной стратифицированной перекрёстной проверки...")

for i, (train, test) in enumerate(cv.split(X_train2, y_train)):
    # Обучение модели
    probas = pipe_lr.fit(X_train2[train], y_train[train]).predict_proba(X_train2[test])
    
    # Вычисление ROC-кривой
    fpr, tpr, thresholds = roc_curve(y_train[test], probas[:, 1], pos_label=1)
    
    # Интерполяция для усреднения
    mean_tpr += np.interp(mean_fpr, fpr, tpr)
    mean_tpr[0] = 0.0
    
    # Вычисление AUC
    roc_auc = auc(fpr, tpr)
    
    print(f'  Складка {i+1}: AUC = {roc_auc:.3f}')
    
    # Построение ROC-кривой для текущей складки
    plt.plot(fpr, tpr, label=f'Складка {i+1} (AUC = {roc_auc:.2f})')

# Усреднение ROC-кривых
mean_tpr /= cv.get_n_splits()
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)

print(f'\nСредний AUC: {mean_auc:.3f}')

# Добавление линий для сравнения
plt.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), 
         label='Случайное угадывание (AUC = 0.5)')

plt.plot(mean_fpr, mean_tpr, 'k--', label=f'Средняя ROC (AUC = {mean_auc:.2f})', lw=2)

plt.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='black', 
         label='Идеальная производительность (AUC = 1.0)')

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Доля ложноположительных (FPR)', fontsize=12)
plt.ylabel('Доля истинно положительных (TPR)', fontsize=12)
plt.title('ROC-кривая для LogisticRegression (2 признака)', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 6. Построение Precision-Recall кривой
print("\n📈 6. ПОСТРОЕНИЕ PRECISION-RECALL КРИВОЙ")
print("-" * 40)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Precision-Recall кривая для каждой складки
ax1 = axes[0]
mean_precision = 0.0
mean_recall = np.linspace(0, 1, 100)

for i, (train, test) in enumerate(cv.split(X_train2, y_train)):
    probas = pipe_lr.fit(X_train2[train], y_train[train]).predict_proba(X_train2[test])
    
    # Вычисление Precision-Recall кривой
    precision, recall, thresholds = precision_recall_curve(y_train[test], probas[:, 1], pos_label=1)
    
    # Интерполяция
    mean_precision += np.interp(mean_recall[::-1], recall[::-1], precision[::-1])[::-1]
    
    # Вычисление Average Precision
    avg_precision = average_precision_score(y_train[test], probas[:, 1])
    
    print(f'  Складка {i+1}: Average Precision = {avg_precision:.3f}')
    
    ax1.plot(recall, precision, label=f'Складка {i+1} (AP = {avg_precision:.2f})')

# Усреднение
mean_precision /= cv.get_n_splits()
mean_ap = average_precision_score(y_train, pipe_lr.fit(X_train2, y_train).predict_proba(X_train2)[:, 1])

print(f'\nСредний Average Precision: {mean_ap:.3f}')

ax1.set_xlabel('Recall', fontsize=12)
ax1.set_ylabel('Precision', fontsize=12)
ax1.set_title('Precision-Recall кривая', fontsize=14, fontweight='bold')
ax1.legend(loc='lower left')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([0, 1])
ax1.set_ylim([0, 1])

# Сравнение с ROC-кривой
ax2 = axes[1]
ax2.plot([0, 1], [0, 1], linestyle='--', color=(0.6, 0.6, 0.6), 
         label='Случайное угадывание (AUC = 0.5)')
ax2.plot(mean_fpr, mean_tpr, 'k--', label=f'Средняя ROC (AUC = {mean_auc:.2f})', lw=2)
ax2.plot([0, 0, 1], [0, 1, 1], linestyle=':', color='black', 
         label='Идеальная производительность (AUC = 1.0)')
ax2.set_xlabel('FPR', fontsize=12)
ax2.set_ylabel('TPR', fontsize=12)
ax2.set_title('ROC-кривая (для сравнения)', fontsize=14, fontweight='bold')
ax2.legend(loc='lower right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim([-0.05, 1.05])
ax2.set_ylim([-0.05, 1.05])

plt.tight_layout()
plt.show()

# 7. ROC-кривая на тестовом наборе
print("\n📈 7. ROC-КРИВАЯ НА ТЕСТОВОМ НАБОРЕ")
print("-" * 40)

# Обучение на полном обучающем наборе
pipe_lr.fit(X_train2, y_train)

# Предсказание вероятностей на тестовом наборе
y_test_proba = pipe_lr.predict_proba(X_test2)[:, 1]

# Вычисление ROC-кривой
fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_proba, pos_label=1)
roc_auc_test = auc(fpr_test, tpr_test)

print(f'ROC AUC на тестовом наборе: {roc_auc_test:.3f}')

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(fpr_test, tpr_test, color='darkorange', lw=2, 
        label=f'Тестовый набор (AUC = {roc_auc_test:.2f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
        label='Случайное угадывание (AUC = 0.5)')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Доля ложноположительных (FPR)', fontsize=12)
ax.set_ylabel('Доля истинно положительных (TPR)', fontsize=12)
ax.set_title('ROC-кривая на тестовом наборе', fontsize=14, fontweight='bold')
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 8. Сравнение с использованием всех признаков
print("\n📊 8. СРАВНЕНИЕ: 2 ПРИЗНАКА VS ВСЕ ПРИЗНАКИ")
print("-" * 40)

# Конвейер со всеми признаками
pipe_lr_all = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=100.0, solver='lbfgs', random_state=1, max_iter=10000)
)

# Обучение и предсказание со всеми признаками
pipe_lr_all.fit(X_train, y_train)
y_test_proba_all = pipe_lr_all.predict_proba(X_test)[:, 1]

# ROC-кривая со всеми признаками
fpr_all, tpr_all, _ = roc_curve(y_test, y_test_proba_all, pos_label=1)
roc_auc_all = auc(fpr_all, tpr_all)

print(f'ROC AUC (2 признака): {roc_auc_test:.3f}')
print(f'ROC AUC (все признаки): {roc_auc_all:.3f}')

improvement = (roc_auc_all - roc_auc_test) * 100
print(f'Улучшение при использовании всех признаков: {improvement:.2f}%')

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(fpr_test, tpr_test, color='blue', lw=2, 
        label=f'2 признака (AUC = {roc_auc_test:.2f})')
ax.plot(fpr_all, tpr_all, color='red', lw=2, 
        label=f'Все признаки (AUC = {roc_auc_all:.2f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
        label='Случайное угадывание (AUC = 0.5)')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Доля ложноположительных (FPR)', fontsize=12)
ax.set_ylabel('Доля истинно положительных (TPR)', fontsize=12)
ax.set_title('Сравнение ROC-кривых: 2 признака vs все признаки', fontsize=14, fontweight='bold')
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 9. Анализ порогов принятия решения
print("\n🔍 9. АНАЛИЗ ПОРОГОВ ПРИНЯТИЯ РЕШЕНИЯ")
print("-" * 40)

# Выбор оптимального порога (максимизация TPR - FPR)
optimal_idx = np.argmax(tpr_test - fpr_test)
optimal_threshold = thresholds_test[optimal_idx]

print(f'Оптимальный порог (максимизация TPR - FPR): {optimal_threshold:.3f}')
print(f'  FPR при оптимальном пороге: {fpr_test[optimal_idx]:.3f}')
print(f'  TPR при оптимальном пороге: {tpr_test[optimal_idx]:.3f}')

# Различные пороги
thresholds_to_test = [0.3, 0.5, 0.7]
print(f'\nАнализ различных порогов:')
for threshold in thresholds_to_test:
    y_pred_threshold = (y_test_proba >= threshold).astype(int)
    tp = np.sum((y_pred_threshold == 1) & (y_test == 1))
    tn = np.sum((y_pred_threshold == 0) & (y_test == 0))
    fp = np.sum((y_pred_threshold == 1) & (y_test == 0))
    fn = np.sum((y_pred_threshold == 0) & (y_test == 1))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    print(f'\n  Порог {threshold}:')
    print(f'    TP={tp}, TN={tn}, FP={fp}, FN={fn}')
    print(f'    Precision: {precision:.3f}')
    print(f'    Recall: {recall:.3f}')

# 10. Визуализация распределения вероятностей
print("\n📈 10. ВИЗУАЛИЗАЦИЯ РАСПРЕДЕЛЕНИЯ ВЕРОЯТНОСТЕЙ")
print("-" * 40)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Распределение вероятностей по классам
ax1 = axes[0]
for label in [0, 1]:
    mask = y_test == label
    ax1.hist(y_test_proba[mask], bins=20, alpha=0.6, 
             label=f'Класс {label} ({le.classes_[label]})', density=True)
ax1.axvline(x=0.5, color='red', linestyle='--', label='Порог 0.5')
ax1.set_xlabel('Предсказанная вероятность класса 1', fontsize=12)
ax1.set_ylabel('Плотность', fontsize=12)
ax1.set_title('Распределение вероятностей по классам', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Boxplot распределения вероятностей
ax2 = axes[1]
data_to_plot = [y_test_proba[y_test == 0], y_test_proba[y_test == 1]]
bp = ax2.boxplot(data_to_plot, tick_labels=['Доброкачественная', 'Злокачественная'],
                 patch_artist=True)
colors_box = ['lightblue', 'lightcoral']
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
ax2.axhline(y=0.5, color='red', linestyle='--', label='Порог 0.5')
ax2.set_ylabel('Предсказанная вероятность класса 1', fontsize=12)
ax2.set_title('Boxplot распределения вероятностей', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# 11. Выводы
print("\n📝 11. ВЫВОДЫ")
print("=" * 60)
print("ROC-кривая позволяет:")
print("  ✅ Оценить производительность классификатора")
print("  ✅ Сравнить различные модели")
print("  ✅ Выбрать оптимальный порог принятия решения")
print("  ✅ Понять компромисс между TPR и FPR")
print("\nROC AUC:")
print("  - 1.0: идеальный классификатор")
print("  - 0.5: случайное угадывание")
print("  - <0.5: хуже случайного")
print("  - Чем ближе к 1.0, тем лучше модель")
print("\nPrecision-Recall кривая:")
print("  - Альтернатива ROC для несбалансированных классов")
print("  - Показывает компромисс между precision и recall")
print("  - Более информативна при редком положительном классе")
print("\nВ данном примере:")
print(f"  - ROC AUC (2 признака): {roc_auc_test:.3f}")
print(f"  - ROC AUC (все признаки): {roc_auc_all:.3f}")
print(f"  - Использование всех признаков улучшило результат на {improvement:.2f}%")
print("\nВыбор порога:")
print("  - Низкий порог: высокий recall, низкий precision")
print("  - Высокий порог: высокий precision, низкий recall")
print("  - Оптимальный порог зависит от задачи")
print("  - В медицине: низкий порог (важно не пропустить болезнь)")
