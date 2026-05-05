# -*- coding: utf-8 -*-
"""
Раздел 6.5.2: Оптимизация правильности и полноты модели классификации

Метрики оценки производительности классификатора:

Основные метрики:
- Error (ERR): (FP + FN) / (FP + FN + TP + TN) - доля ошибок
- Accuracy (ACC): 1 - ERR = (TP + TN) / (FP + FN + TP + TN) - правильность

Метрики для несбалансированных классов:
- False Positive Rate (FPR): FP / (FP + TN) - доля ложноположительных
- True Positive Rate (TPR): TP / (FN + TP) - доля истинно положительных

Метрики точности и полноты:
- Precision (PRE): TP / (TP + FP) - точность предсказания положительного класса
- Recall (REC): TP / (FN + TP) = TPR - полнота обнаружения положительного класса
- F1-score: 2 * (PRE * REC) / (PRE + REC) - гармоническое среднее precision и recall

Комплексная метрика:
- Matthews Correlation Coefficient (MCC): учитывает все элементы матрицы несоответствий
  Диапазон: [-1, 1], где 1 - идеальное предсказание, 0 - случайное, -1 - обратное

Компромисс между Precision и Recall:
- Высокий Recall: минимизирует FN (пропуск положительных)
- Высокий Precision: минимизирует FP (ложные срабатывания)
- F1-score: баланс между precision и recall
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
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             matthews_corrcoef, confusion_matrix, 
                             make_scorer)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline

print("📊 ОПТИМИЗАЦИЯ ПРАВИЛЬНОСТИ И ПОЛНОТЫ МОДЕЛИ")
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
y_pred = pipe_svc.predict(X_test)

print("Модель SVM обучена и предсказания выполнены")

# 4. Вычисление матрицы несоответствий
print("\n📊 4. МАТРИЦА НЕСООТВЕТСТВИЙ")
print("-" * 40)

confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print(confmat)

tn, fp, fn, tp = confmat.ravel()

print(f'\nTN={tn}, FP={fp}, FN={fn}, TP={tp}')

# 5. Вычисление основных метрик вручную
print("\n📈 5. ВЫЧИСЛЕНИЕ МЕТРИК ВРУЧНУЮ")
print("-" * 40)

# Error и Accuracy
err = (fp + fn) / (fp + fn + tp + tn)
acc = 1 - err
print(f'Error (ERR): {err:.4f}')
print(f'  = (FP + FN) / (FP + FN + TP + TN)')
print(f'  = ({fp} + {fn}) / ({fp} + {fn} + {tp} + {tn})')

print(f'\nAccuracy (ACC): {acc:.4f}')
print(f'  = 1 - ERR = (TP + TN) / (FP + FN + TP + TN)')
print(f'  = ({tp} + {tn}) / ({fp} + {fn} + {tp} + {tn})')

# FPR и TPR
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
tpr = tp / (fn + tp) if (fn + tp) > 0 else 0
print(f'\nFalse Positive Rate (FPR): {fpr:.4f}')
print(f'  = FP / (FP + TN) = {fp} / ({fp} + {tn})')

print(f'\nTrue Positive Rate (TPR): {tpr:.4f}')
print(f'  = TP / (FN + TP) = {tp} / ({fn} + {tp})')

# Precision, Recall, F1
pre = tp / (tp + fp) if (tp + fp) > 0 else 0
rec = tp / (fn + tp) if (fn + tp) > 0 else 0
f1 = 2 * (pre * rec) / (pre + rec) if (pre + rec) > 0 else 0

print(f'\nPrecision (PRE): {pre:.4f}')
print(f'  = TP / (TP + FP) = {tp} / ({tp} + {fp})')

print(f'\nRecall (REC): {rec:.4f}')
print(f'  = TP / (FN + TP) = {tp} / ({fn} + {tp})')

print(f'\nF1-score: {f1:.4f}')
print(f'  = 2 * (PRE * REC) / (PRE + REC)')
print(f'  = 2 * ({pre:.4f} * {rec:.4f}) / ({pre:.4f} + {rec:.4f})')

# 6. Вычисление метрик с помощью sklearn
print("\n📊 6. ВЫЧИСЛЕНИЕ МЕТРИК С ПОМОЩЬЮ SKLEARN")
print("-" * 40)

pre_val = precision_score(y_true=y_test, y_pred=y_pred)
rec_val = recall_score(y_true=y_test, y_pred=y_pred)
f1_val = f1_score(y_true=y_test, y_pred=y_pred)
mcc_val = matthews_corrcoef(y_true=y_test, y_pred=y_pred)

print(f'Precision: {pre_val:.4f}')
print(f'Recall: {rec_val:.4f}')
print(f'F1-score: {f1_val:.4f}')
print(f'Matthews Correlation Coefficient (MCC): {mcc_val:.4f}')

print(f'\nСравнение с ручными вычислениями:')
print(f'  Precision: {pre:.4f} vs {pre_val:.4f} {"✅" if abs(pre - pre_val) < 0.001 else "❌"}')
print(f'  Recall: {rec:.4f} vs {rec_val:.4f} {"✅" if abs(rec - rec_val) < 0.001 else "❌"}')
print(f'  F1-score: {f1:.4f} vs {f1_val:.4f} {"✅" if abs(f1 - f1_val) < 0.001 else "❌"}')

# 7. Интерпретация метрик в контексте медицинской диагностики
print("\n🏥 7. ИНТЕРПРЕТАЦИЯ В МЕДИЦИНСКОЙ ДИАГНОСТИКЕ")
print("-" * 40)

print("Контекст: диагностика злокачественных опухолей")
print(f'\nPrecision ({pre_val:.4f}):')
print("  - Из всех предсказанных как злокачественные,")
print(f"  - {pre_val*100:.1f}% действительно злокачественные")
print("  - Высокий precision → меньше ложных тревог (FP)")
print("  - Важно для избежания лишних процедур")

print(f'\nRecall ({rec_val:.4f}):')
print("  - Из всех реально злокачественных,")
print(f"  - {rec_val*100:.1f}% были обнаружены")
print("  - Высокий recall → меньше пропущенных случаев (FN)")
print("  - Критически важно для раннего лечения")

print(f'\nF1-score ({f1_val:.4f}):')
print("  - Гармоническое среднее precision и recall")
print("  - Баланс между точностью и полнотой")
print("  - Полезен при несбалансированных классах")

print(f'\nMCC ({mcc_val:.4f}):')
print("  - Учитывает все элементы матрицы несоответствий")
print("  - Диапазон: [-1, 1]")
print("  - 1 = идеальное предсказание")
print("  - 0 = случайное угадывание")
print("  - -1 = обратное предсказание")
print("  - Считается лучшей метрикой для бинарной классификации")

# 8. Компромисс между Precision и Recall
print("\n⚖️ 8. КОМПРОМИСС МЕЖДУ PRECISION И RECALL")
print("-" * 40)

print("Оптимизация по Recall:")
print("  - Минимизирует FN (пропуск злокачественных)")
print("  - Но увеличивает FP (ложные тревоги)")
print("  - Результат: больше пациентов получат лечение,")
print("    но некоторые будут лечиться напрасно")

print("\nОптимизация по Precision:")
print("  - Минимизирует FP (ложные тревоги)")
print("  - Но увеличивает FN (пропуск злокачественных)")
print("  - Результат: меньше ложных тревог,")
print("    но некоторые пациенты пропустят лечение")

print("\nF1-score:")
print("  - Балансирует precision и recall")
print("  - Полезен, когда оба параметра важны")
print("  - Оптимален для большинства практических задач")

# 9. GridSearchCV с F1-score
print("\n🔍 9. GRIDSEARCHCV С F1-SCORE")
print("-" * 40)

c_gamma_range = [0.01, 0.1, 1.0, 10.0]

param_grid = [
    {
        'svc__C': c_gamma_range,
        'svc__kernel': ['linear']
    },
    {
        'svc__C': c_gamma_range,
        'svc__gamma': c_gamma_range,
        'svc__kernel': ['rbf']
    }
]

print("Сетка гиперпараметров:")
print(f"  - C и gamma: {c_gamma_range}")
print("  - Ядра: ['linear', 'rbf']")

# Создание scorer для F1-score
scorer = make_scorer(f1_score, pos_label=1)

print("\nИспользование F1-score в качестве метрики:")
print("  - scorer = make_scorer(f1_score, pos_label=1)")
print("  - pos_label=1 указывает, что класс 1 (злокачественная) - положительный")

gs = GridSearchCV(
    estimator=pipe_svc,
    param_grid=param_grid,
    scoring=scorer,
    cv=10
)

print("\nЗапуск GridSearchCV с F1-score...")
gs.fit(X_train, y_train)

print(f'\nЛучший F1-score (CV): {gs.best_score_:.4f}')
print(f'Лучшие параметры: {gs.best_params_}')

# 10. Сравнение с accuracy-based GridSearch
print("\n📊 10. СРАВНЕНИЕ С ACCURACY-BASED GRIDSEARCH")
print("-" * 40)

gs_acc = GridSearchCV(
    estimator=pipe_svc,
    param_grid=param_grid,
    scoring='accuracy',
    cv=10
)

gs_acc.fit(X_train, y_train)

print(f'GridSearch с Accuracy:')
print(f'  Лучший Accuracy (CV): {gs_acc.best_score_:.4f}')
print(f'  Лучшие параметры: {gs_acc.best_params_}')

print(f'\nGridSearch с F1-score:')
print(f'  Лучший F1-score (CV): {gs.best_score_:.4f}')
print(f'  Лучшие параметры: {gs.best_params_}')

# Оценка на тестовом наборе
y_pred_acc = gs_acc.best_estimator_.predict(X_test)
y_pred_f1 = gs.best_estimator_.predict(X_test)

f1_acc = f1_score(y_test, y_pred_acc)
f1_f1 = f1_score(y_test, y_pred_f1)

print(f'\nТестовый набор:')
print(f'  Accuracy-based модель: F1 = {f1_acc:.4f}')
print(f'  F1-score-based модель: F1 = {f1_f1:.4f}')

if f1_f1 > f1_acc:
    improvement = (f1_f1 - f1_acc) * 100
    print(f'\n✅ Оптимизация по F1-score улучшила F1 на {improvement:.2f}%')
else:
    degradation = (f1_acc - f1_f1) * 100
    print(f'\n⚠️  Оптимизация по Accuracy дала лучший F1 на {degradation:.2f}%')

# 11. Визуализация метрик
print("\n📈 11. ВИЗУАЛИЗАЦИЯ МЕТРИК")
print("-" * 40)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# График 1: Сравнение метрик
ax1 = axes[0, 0]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'MCC']
values = [acc, pre_val, rec_val, f1_val, mcc_val]
colors = ['blue', 'green', 'orange', 'red', 'purple']
bars = ax1.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
ax1.set_ylabel('Значение')
ax1.set_title('Сравнение метрик оценки')
ax1.set_ylim([0.8, 1.0])
ax1.grid(True, alpha=0.3, axis='y')
# Добавление значений на столбцы
for bar, val in zip(bars, values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', va='bottom', fontsize=10)

# График 2: Precision vs Recall
ax2 = axes[0, 1]
ax2.scatter([rec_val], [pre_val], s=200, color='red', marker='o', 
            label=f'Текущая модель\n(Precision={pre_val:.3f}, Recall={rec_val:.3f})')
ax2.set_xlabel('Recall')
ax2.set_ylabel('Precision')
ax2.set_title('Precision vs Recall')
ax2.set_xlim([0.9, 1.0])
ax2.set_ylim([0.9, 1.0])
ax2.grid(True, alpha=0.3)
ax2.legend()
# Добавление идеальной точки
ax2.scatter([1.0], [1.0], s=100, color='green', marker='*', label='Идеал (1, 1)')
ax2.legend()

# График 3: Матрица несоответствий с процентами
ax3 = axes[1, 0]
confmat_norm = confmat.astype('float') / confmat.sum(axis=1)[:, np.newaxis]
im = ax3.imshow(confmat, cmap=plt.cm.Blues, alpha=0.7)
cbar = plt.colorbar(im, ax=ax3)
cbar.set_label('Количество', rotation=270, labelpad=20)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        text = f'{confmat[i, j]}\n({confmat_norm[i, j]*100:.1f}%)'
        ax3.text(x=j, y=i, s=text, va='center', ha='center',
                fontsize=11, fontweight='bold',
                color='white' if confmat[i, j] > confmat.max() / 2 else 'black')
ax3.xaxis.set_ticks_position('bottom')
ax3.set_xticks([0, 1])
ax3.set_yticks([0, 1])
ax3.set_xticklabels(['Доброкачественная', 'Злокачественная'])
ax3.set_yticklabels(['Доброкачественная', 'Злокачественная'])
ax3.set_xlabel('Предсказанная метка')
ax3.set_ylabel('Истинная метка')
ax3.set_title('Матрица несоответствий с %')

# График 4: Сравнение моделей
ax4 = axes[1, 1]
models = ['Accuracy-based', 'F1-score-based']
f1_values = [f1_acc, f1_f1]
bars = ax4.bar(models, f1_values, color=['lightblue', 'lightcoral'], 
               alpha=0.7, edgecolor='black')
ax4.set_ylabel('F1-score')
ax4.set_title('Сравнение моделей на тестовом наборе')
ax4.set_ylim([0.9, 1.0])
ax4.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, f1_values):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{val:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

# 12. Выводы
print("\n📝 12. ВЫВОДЫ")
print("=" * 60)
print("Метрики оценки производительности:")
print("  ✅ Accuracy: общая правильность классификации")
print("  ✅ Precision: точность предсказания положительного класса")
print("  ✅ Recall: полнота обнаружения положительного класса")
print("  ✅ F1-score: баланс между precision и recall")
print("  ✅ MCC: комплексная метрика, учитывающая все элементы")
print("\nВыбор метрики:")
print("  - Accuracy: подходит для сбалансированных классов")
print("  - Precision: важна, когда FP дорогостоящи")
print("  - Recall: важна, когда FN критичны (медицина)")
print("  - F1-score: баланс между precision и recall")
print("  - MCC: лучшая метрика для бинарной классификации")
print("\nВ медицинской диагностике:")
print("  - Recall критичнее (пропуск болезни опаснее)")
print("  - Но важен баланс с precision (избегать лишних процедур)")
print("  - F1-score или MCC - хорошие компромиссные метрики")
print("  - GridSearchCV с F1-score может дать лучшие результаты")
