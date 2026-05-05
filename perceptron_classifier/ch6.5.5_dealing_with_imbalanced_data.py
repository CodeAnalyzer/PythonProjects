# -*- coding: utf-8 -*-
"""
Раздел 6.5.5: Борьба с дисбалансом классов

Дисбаланс классов - ситуация, когда экземпляры одного или нескольких классов
представлены в наборе данных слишком часто. Это распространённая проблема в
многих областях: фильтрация спама, обнаружение мошенничества, диагностика болезней.

Проблема дисбаланса:
- Модель может достичь высокой точности, просто предсказывая доминирующий класс
- Accuracy не является информативной метрикой
- Алгоритмы ML оптимизируют функцию потерь на всех примерах, что приводит к смещению

Методы борьбы с дисбалансом:
1. Выбор правильных метрик (precision, recall, F1, ROC AUC вместо accuracy)
2. class_weight='balanced' - взвешивание классов при обучении
3. Upsampling (повышение дискретизации) - увеличение миноритарного класса
4. Downsampling (понижение дискретизации) - уменьшение доминирующего класса
5. SMOTE (Synthetic Minority Over-sampling Technique) - создание синтетических примеров
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
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_curve, auc)
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.utils import resample
from sklearn.pipeline import make_pipeline

print("⚖️ БОРЬБА С ДИСБАЛАНСОМ КЛАССОВ")
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

print(f'Исходный набор данных:')
print(f'  Класс 0 (доброкачественная): {np.sum(y == 0)} образцов')
print(f'  Класс 1 (злокачественная): {np.sum(y == 1)} образцов')
print(f'  Всего: {len(y)} образцов')
print(f'  Соотношение классов: {np.sum(y == 1) / np.sum(y == 0):.2f}')

# 2. Создание несбалансированного набора данных
print("\n🔀 2. СОЗДАНИЕ НЕСБАЛАНСИРОВАННОГО НАБОРА ДАННЫХ")
print("-" * 40)

# Берём все доброкачественные (класс 0) и только 40 злокачественных (класс 1)
X_imb = np.vstack((X[y == 0], X[y == 1][:40]))
y_imb = np.hstack((y[y == 0], y[y == 1][:40]))

print(f'Несбалансированный набор данных:')
print(f'  Класс 0 (доброкачественная): {np.sum(y_imb == 0)} образцов')
print(f'  Класс 1 (злокачественная): {np.sum(y_imb == 1)} образцов')
print(f'  Всего: {len(y_imb)} образцов')
print(f'  Соотношение классов: {np.sum(y_imb == 1) / np.sum(y_imb == 0):.2f}')

# 3. Базовый классификатор (всегда предсказывает доминирующий класс)
print("\n🎯 3. БАЗОВЫЙ КЛАССИФИКАТОР (ВСЕГДА ДОБРОКАЧЕСТВЕННАЯ)")
print("-" * 40)

y_pred_zero = np.zeros(y_imb.shape[0])
accuracy_zero = np.mean(y_pred_zero == y_imb) * 100

print(f'Классификатор, всегда предсказывающий класс 0:')
print(f'  Точность: {accuracy_zero:.2f}%')
print(f'\n⚠️  Это показывает, что accuracy не информативна при дисбалансе!')
print(f'    Модель достигает {accuracy_zero:.1f}% точности без обучения!')

# 4. Разделение на обучающие и тестовые наборы
print("\n🔀 4. РАЗДЕЛЕНИЕ ДАННЫХ")
print("-" * 40)

X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
    X_imb, y_imb, test_size=0.2, random_state=1, stratify=y_imb
)

print(f'Обучающий набор: {X_train_imb.shape[0]} образцов')
print(f'  Класс 0: {np.sum(y_train_imb == 0)}')
print(f'  Класс 1: {np.sum(y_train_imb == 1)}')
print(f'Тестовый набор: {X_test_imb.shape[0]} образцов')
print(f'  Класс 0: {np.sum(y_test_imb == 0)}')
print(f'  Класс 1: {np.sum(y_test_imb == 1)}')

# 5. Логистическая регрессия без учёта дисбаланса
print("\n🔧 5. ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ БЕЗ УЧЁТА ДИСБАЛАНСА")
print("-" * 40)

pipe_lr = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=1, max_iter=10000)
)

pipe_lr.fit(X_train_imb, y_train_imb)
y_pred_imb = pipe_lr.predict(X_test_imb)

acc_imb = accuracy_score(y_test_imb, y_pred_imb)
pre_imb = precision_score(y_test_imb, y_pred_imb)
rec_imb = recall_score(y_test_imb, y_pred_imb)
f1_imb = f1_score(y_test_imb, y_pred_imb)

print(f'Результаты на несбалансированном наборе:')
print(f'  Accuracy: {acc_imb:.4f}')
print(f'  Precision: {pre_imb:.4f}')
print(f'  Recall: {rec_imb:.4f}')
print(f'  F1-score: {f1_imb:.4f}')

confmat_imb = confusion_matrix(y_test_imb, y_pred_imb)
print(f'\nМатрица несоответствий:')
print(confmat_imb)

# 6. Логистическая регрессия с class_weight='balanced'
print("\n⚖️ 6. ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ С CLASS_WEIGHT='BALANCED'")
print("-" * 40)

pipe_lr_balanced = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight='balanced', random_state=1, max_iter=10000)
)

pipe_lr_balanced.fit(X_train_imb, y_train_imb)
y_pred_balanced = pipe_lr_balanced.predict(X_test_imb)

acc_balanced = accuracy_score(y_test_imb, y_pred_balanced)
pre_balanced = precision_score(y_test_imb, y_pred_balanced)
rec_balanced = recall_score(y_test_imb, y_pred_balanced)
f1_balanced = f1_score(y_test_imb, y_pred_balanced)

print(f'Результаты с class_weight="balanced":')
print(f'  Accuracy: {acc_balanced:.4f}')
print(f'  Precision: {pre_balanced:.4f}')
print(f'  Recall: {rec_balanced:.4f}')
print(f'  F1-score: {f1_balanced:.4f}')

confmat_balanced = confusion_matrix(y_test_imb, y_pred_balanced)
print(f'\nМатрица несоответствий:')
print(confmat_balanced)

print(f'\nСравнение с обычной моделью:')
print(f'  Accuracy: {acc_imb:.4f} → {acc_balanced:.4f} ({(acc_balanced - acc_imb)*100:+.2f}%)')
print(f'  Precision: {pre_imb:.4f} → {pre_balanced:.4f} ({(pre_balanced - pre_imb)*100:+.2f}%)')
print(f'  Recall: {rec_imb:.4f} → {rec_balanced:.4f} ({(rec_balanced - rec_imb)*100:+.2f}%)')
print(f'  F1-score: {f1_imb:.4f} → {f1_balanced:.4f} ({(f1_balanced - f1_imb)*100:+.2f}%)')

# 7. Upsampling (повышение дискретизации) миноритарного класса
print("\n📈 7. UPSAMPLING (ПОВЫШЕНИЕ ДИСКРЕТИЗАЦИИ)")
print("-" * 40)

print(f'Количество экземпляров класса 1 до upsampling: {X_imb[y_imb == 1].shape[0]}')

# Upsampling класса 1 до количества класса 0
X_upsampled, y_upsampled = resample(
    X_imb[y_imb == 1],
    y_imb[y_imb == 1],
    replace=True,
    n_samples=X_imb[y_imb == 0].shape[0],
    random_state=123
)

print(f'Количество экземпляров класса 1 после upsampling: {X_upsampled.shape[0]}')

# Объединение с классом 0
X_bal = np.vstack((X_imb[y_imb == 0], X_upsampled))
y_bal = np.hstack((y_imb[y_imb == 0], y_upsampled))

print(f'\nСбалансированный набор данных:')
print(f'  Класс 0: {np.sum(y_bal == 0)} образцов')
print(f'  Класс 1: {np.sum(y_bal == 1)} образцов')
print(f'  Всего: {len(y_bal)} образцов')

# Проверка базового классификатора
y_pred_zero_bal = np.zeros(y_bal.shape[0])
accuracy_zero_bal = np.mean(y_pred_zero_bal == y_bal) * 100
print(f'\nБазовый классификатор на сбалансированном наборе:')
print(f'  Точность: {accuracy_zero_bal:.2f}% (было {accuracy_zero:.2f}%)')

# Разделение и обучение на сбалансированном наборе
X_train_bal, X_test_bal, y_train_bal, y_test_bal = train_test_split(
    X_bal, y_bal, test_size=0.2, random_state=1, stratify=y_bal
)

pipe_lr_upsampled = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=1, max_iter=10000)
)

pipe_lr_upsampled.fit(X_train_bal, y_train_bal)
y_pred_upsampled = pipe_lr_upsampled.predict(X_test_bal)

acc_upsampled = accuracy_score(y_test_bal, y_pred_upsampled)
pre_upsampled = precision_score(y_test_bal, y_pred_upsampled)
rec_upsampled = recall_score(y_test_bal, y_pred_upsampled)
f1_upsampled = f1_score(y_test_bal, y_pred_upsampled)

print(f'\nРезультаты на сбалансированном наборе (upsampling):')
print(f'  Accuracy: {acc_upsampled:.4f}')
print(f'  Precision: {pre_upsampled:.4f}')
print(f'  Recall: {rec_upsampled:.4f}')
print(f'  F1-score: {f1_upsampled:.4f}')

confmat_upsampled = confusion_matrix(y_test_bal, y_pred_upsampled)
print(f'\nМатрица несоответствий:')
print(confmat_upsampled)

# 8. Downsampling (понижение дискретизации) доминирующего класса
print("\n📉 8. DOWNSAMPLING (ПОНИЖЕНИЕ ДИСКРЕТИЗАЦИИ)")
print("-" * 40)

print(f'Количество экземпляров класса 0 до downsampling: {X_imb[y_imb == 0].shape[0]}')

# Downsampling класса 0 до количества класса 1
X_downsampled, y_downsampled = resample(
    X_imb[y_imb == 0],
    y_imb[y_imb == 0],
    replace=False,
    n_samples=X_imb[y_imb == 1].shape[0],
    random_state=123
)

print(f'Количество экземпляров класса 0 после downsampling: {X_downsampled.shape[0]}')

# Объединение с классом 1
X_bal_down = np.vstack((X_downsampled, X_imb[y_imb == 1]))
y_bal_down = np.hstack((y_downsampled, y_imb[y_imb == 1]))

print(f'\nСбалансированный набор данных (downsampling):')
print(f'  Класс 0: {np.sum(y_bal_down == 0)} образцов')
print(f'  Класс 1: {np.sum(y_bal_down == 1)} образцов')
print(f'  Всего: {len(y_bal_down)} образцов')

# Разделение и обучение
X_train_down, X_test_down, y_train_down, y_test_down = train_test_split(
    X_bal_down, y_bal_down, test_size=0.2, random_state=1, stratify=y_bal_down
)

pipe_lr_downsampled = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=1, max_iter=10000)
)

pipe_lr_downsampled.fit(X_train_down, y_train_down)
y_pred_downsampled = pipe_lr_downsampled.predict(X_test_down)

acc_downsampled = accuracy_score(y_test_down, y_pred_downsampled)
pre_downsampled = precision_score(y_test_down, y_pred_downsampled)
rec_downsampled = recall_score(y_test_down, y_pred_downsampled)
f1_downsampled = f1_score(y_test_down, y_pred_downsampled)

print(f'\nРезультаты на сбалансированном наборе (downsampling):')
print(f'  Accuracy: {acc_downsampled:.4f}')
print(f'  Precision: {pre_downsampled:.4f}')
print(f'  Recall: {rec_downsampled:.4f}')
print(f'  F1-score: {f1_downsampled:.4f}')

confmat_downsampled = confusion_matrix(y_test_down, y_pred_downsampled)
print(f'\nМатрица несоответствий:')
print(confmat_downsampled)

# 9. Сравнение всех подходов
print("\n📊 9. СРАВНЕНИЕ ВСЕХ ПОДХОДОВ")
print("-" * 40)

approaches = [
    'Без учёта дисбаланса',
    'class_weight="balanced"',
    'Upsampling',
    'Downsampling'
]

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-score']
results = [
    [acc_imb, pre_imb, rec_imb, f1_imb],
    [acc_balanced, pre_balanced, rec_balanced, f1_balanced],
    [acc_upsampled, pre_upsampled, rec_upsampled, f1_upsampled],
    [acc_downsampled, pre_downsampled, rec_downsampled, f1_downsampled]
]

print(f'{"Подход":<30} {"Accuracy":<10} {"Precision":<10} {"Recall":<10} {"F1-score":<10}')
print('-' * 70)
for approach, result in zip(approaches, results):
    print(f'{approach:<30} {result[0]:<10.4f} {result[1]:<10.4f} {result[2]:<10.4f} {result[3]:<10.4f}')

# 10. Визуализация результатов
print("\n📈 10. ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# График 1: Сравнение метрик
ax1 = axes[0, 0]
x = np.arange(len(approaches))
width = 0.2
for i, metric in enumerate(metrics):
    values = [result[i] for result in results]
    offset = (i - 1.5) * width
    ax1.bar(x + offset, values, width, label=metric)
ax1.set_xlabel('Подход')
ax1.set_ylabel('Значение')
ax1.set_title('Сравнение метрик по подходам')
ax1.set_xticks(x)
ax1.set_xticklabels(approaches, rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim([0, 1])

# График 2: Recall vs Precision
ax2 = axes[0, 1]
for i, approach in enumerate(approaches):
    ax2.scatter(results[i][2], results[i][1], s=100, label=approach)
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Recall vs Precision')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_xlim([0, 1])
ax2.set_ylim([0, 1])

# График 3: Распределение классов (до и после)
ax3 = axes[1, 0]
before = [np.sum(y_imb == 0), np.sum(y_imb == 1)]
after_up = [np.sum(y_bal == 0), np.sum(y_bal == 1)]
after_down = [np.sum(y_bal_down == 0), np.sum(y_bal_down == 1)]

x = np.arange(2)
width = 0.25
ax3.bar(x - width, before, width, label='Исходный', color='lightcoral')
ax3.bar(x, after_up, width, label='После upsampling', color='lightblue')
ax3.bar(x + width, after_down, width, label='После downsampling', color='lightgreen')
ax3.set_xlabel('Класс')
ax3.set_ylabel('Количество образцов')
ax3.set_title('Распределение классов')
ax3.set_xticks(x)
ax3.set_xticklabels(['Класс 0', 'Класс 1'])
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# График 4: F1-score по подходам
ax4 = axes[1, 1]
f1_values = [result[3] for result in results]
bars = ax4.bar(approaches, f1_values, color=['lightcoral', 'lightblue', 'lightgreen', 'orange'], 
               alpha=0.7, edgecolor='black')
ax4.set_ylabel('F1-score')
ax4.set_title('F1-score по подходам')
ax4.set_xticks(range(len(approaches)))
ax4.set_xticklabels(approaches, rotation=45, ha='right')
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim([0, 1])
# Добавление значений
for bar, val in zip(bars, f1_values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# 11. ROC-кривые для разных подходов
print("\n📈 11. ROC-КРИВЫЕ ДЛЯ РАЗНЫХ ПОДХОДОВ")
print("-" * 40)

fig, ax = plt.subplots(figsize=(8, 6))

# Без учёта дисбаланса
y_proba_imb = pipe_lr.predict_proba(X_test_imb)[:, 1]
fpr_imb, tpr_imb, _ = roc_curve(y_test_imb, y_proba_imb)
roc_auc_imb = auc(fpr_imb, tpr_imb)
ax.plot(fpr_imb, tpr_imb, label=f'Без учёта дисбаланса (AUC = {roc_auc_imb:.2f})')

# class_weight='balanced'
y_proba_balanced = pipe_lr_balanced.predict_proba(X_test_imb)[:, 1]
fpr_balanced, tpr_balanced, _ = roc_curve(y_test_imb, y_proba_balanced)
roc_auc_balanced = auc(fpr_balanced, tpr_balanced)
ax.plot(fpr_balanced, tpr_balanced, label=f'class_weight="balanced" (AUC = {roc_auc_balanced:.2f})')

# Upsampling
y_proba_upsampled = pipe_lr_upsampled.predict_proba(X_test_bal)[:, 1]
fpr_upsampled, tpr_upsampled, _ = roc_curve(y_test_bal, y_proba_upsampled)
roc_auc_upsampled = auc(fpr_upsampled, tpr_upsampled)
ax.plot(fpr_upsampled, tpr_upsampled, label=f'Upsampling (AUC = {roc_auc_upsampled:.2f})')

# Downsampling
y_proba_downsampled = pipe_lr_downsampled.predict_proba(X_test_down)[:, 1]
fpr_downsampled, tpr_downsampled, _ = roc_curve(y_test_down, y_proba_downsampled)
roc_auc_downsampled = auc(fpr_downsampled, tpr_downsampled)
ax.plot(fpr_downsampled, tpr_downsampled, label=f'Downsampling (AUC = {roc_auc_downsampled:.2f})')

ax.plot([0, 1], [0, 1], 'k--', label='Случайное угадывание (AUC = 0.5)')
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
ax.set_title('ROC-кривые для разных подходов')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 12. Выводы
print("\n📝 12. ВЫВОДЫ")
print("=" * 60)
print("Проблема дисбаланса классов:")
print("  ⚠️  Accuracy не информативна при дисбалансе")
print("  ⚠️  Модель может достигать высокой точности без обучения")
print("  ⚠️  Алгоритмы ML смещаются к доминирующему классу")
print("\nМетоды борьбы с дисбалансом:")
print("  ✅ class_weight='balanced': взвешивание классов при обучении")
print("  ✅ Upsampling: увеличение миноритарного класса")
print("  ✅ Downsampling: уменьшение доминирующего класса")
print("  ✅ SMOTE: создание синтетических примеров (не реализовано)")
print("\nВыбор метрик:")
print("  - При дисбалансе используйте precision, recall, F1, ROC AUC")
print("  - Recall важен, когда FN критичны (медицина)")
print("  - Precision важен, когда FP дорогостоящи (спам)")
print("  - F1-score балансирует precision и recall")
print("\nРекомендации:")
print("  - Всегда анализируйте распределение классов")
print("  - Используйте соответствующие метрики")
print("  - Пробуйте различные методы борьбы с дисбалансом")
print("  - Оценивайте результаты и выбирайте лучший подход")
print("  - Не существует универсального решения для всех задач")
