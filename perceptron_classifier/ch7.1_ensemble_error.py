# -*- coding: utf-8 -*-
"""
Раздел 7.1: Обучение ансамблей моделей

Ансамблевые методы (ensemble methods) - это подход, при котором различные
классификаторы объединяются в метаклассификатор с лучшей обобщающей способностью,
чем каждый классификатор в отдельности.

Принцип работы ансамблей:
- Если базовые классификаторы работают лучше случайного угадывания (E < 0.5)
- И ошибки классификаторов независимы (некоррелированы)
- То ансамбль будет иметь меньшую ошибку, чем каждый классификатор отдельно

Теоретическое обоснование:
- Для n классификаторов с ошибкой E
- Ансамбль ошибается, если > n/2 классификаторов ошибаются
- Используется биномиальное распределение для расчёта вероятности

Формула ансамблевой ошибки:
ensemble_error = sum(comb(n, k) * E^k * (1-E)^(n-k) for k in range(ceil(n/2), n+1))

Где:
- n: количество классификаторов
- E: базовая ошибка каждого классификатора
- k: количество классификаторов, совершивших ошибку
"""
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb
import math

print("🎯 ОБУЧЕНИЕ АНСАМБЛЕЙ МОДЕЛЕЙ")
print("=" * 60)

# 1. Функция для вычисления ансамблевой ошибки
print("\n📊 1. ФУНКЦИЯ ВЫЧИСЛЕНИЯ АНСАМБЛЕВОЙ ОШИБКИ")
print("-" * 40)

def ensemble_error(n_classifier, error):
    """
    Вычисляет вероятность ошибки ансамбля из n_classifier классификаторов,
    каждый из которых имеет базовую ошибку error.
    
    Параметры:
    -----------
    n_classifier : int
        Количество классификаторов в ансамбле
    error : float
        Базовая ошибка каждого классификатора (0 <= error <= 1)
    
    Возвращает:
    -----------
    float
        Вероятность ошибки ансамбля
    """
    # Ансамбль ошибается, если большинство классификаторов ошибаются
    # Для нечётного n: > n/2
    # Для чётного n: >= n/2 (рассматривается как ошибка)
    k_start = int(math.ceil(n_classifier / 2.))
    
    # Вычисляем вероятность того, что k или более классификаторов ошибаются
    # Используем биномиальное распределение
    probs = [comb(n_classifier, k) * error**k * (1-error)**(n_classifier - k)
             for k in range(k_start, n_classifier + 1)]
    
    return sum(probs)

print("Функция ensemble_error(n_classifier, error):")
print("  - Вычисляет вероятность ошибки ансамбля")
print("  - Использует биномиальное распределение")
print("  - Ансамбль ошибается, если > n/2 классификаторов ошибаются")

# 2. Пример из книги: 11 классификаторов с ошибкой 0.25
print("\n📈 2. ПРИМЕР ИЗ КНИГИ")
print("-" * 40)

n_classifier = 11
base_error = 0.25

ens_error = ensemble_error(n_classifier=n_classifier, error=base_error)

print(f'Количество классификаторов: {n_classifier}')
print(f'Базовая ошибка каждого классификатора: {base_error}')
print(f'Ансамблевая ошибка: {ens_error:.6f}')
print(f'\nУлучшение: {(base_error - ens_error) / base_error * 100:.2f}%')

print(f'\nОбъяснение:')
print(f'  - Для принятия решения нужно majority vote (> {n_classifier/2:.0f} классификаторов)')
print(f'  - Ансамбль ошибается, если {math.ceil(n_classifier/2.)} или более классификаторов ошибаются')
print(f'  - Вероятность этого события: {ens_error:.6f}')

# 3. Вычисление ансамблевой ошибки для диапазона базовых ошибок
print("\n📊 3. ВЫЧИСЛЕНИЕ ДЛЯ ДИАПАЗОНА ОШИБОК")
print("-" * 40)

error_range = np.arange(0.0, 1.01, 0.01)
ens_errors = [ensemble_error(n_classifier=n_classifier, error=error) 
              for error in error_range]

print(f'Диапазон базовых ошибок: [{error_range[0]:.2f}, {error_range[-1]:.2f}]')
print(f'Количество точек: {len(error_range)}')

# Покажем несколько примеров
examples = [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0]
print(f'\nПримеры для n={n_classifier}:')
for error in examples:
    ens_err = ensemble_error(n_classifier=n_classifier, error=error)
    improvement = (error - ens_err) if error > 0 else 0
    print(f'  Базовая ошибка={error:.2f} → Ансамблевая ошибка={ens_err:.4f} (улучшение={improvement:.4f})')

# 4. Визуализация зависимости ансамблевой ошибки от базовой
print("\n📈 4. ВИЗУАЛИЗАЦИЯ ЗАВИСИМОСТИ")
print("-" * 40)

plt.figure(figsize=(10, 6))
plt.plot(error_range, ens_errors, label='Ансамблевая ошибка', linewidth=2, color='blue')
plt.plot(error_range, error_range, linestyle='--', label='Базовая ошибка', linewidth=2, color='red')
plt.xlabel('Базовая ошибка', fontsize=12)
plt.ylabel('Базовая/ансамблевая ошибка', fontsize=12)
plt.title(f'Зависимость ансамблевой ошибки от базовой (n={n_classifier})', 
          fontsize=14, fontweight='bold')
plt.legend(loc='upper left', fontsize=10)
plt.grid(alpha=0.5)
plt.xlim([0, 1])
plt.ylim([0, 1])

# Добавляем точки для key значений
key_errors = [0.25, 0.5, 0.75]
for error in key_errors:
    ens_err = ensemble_error(n_classifier=n_classifier, error=error)
    plt.scatter([error], [ens_err], s=100, color='green', zorder=5)
    plt.annotate(f'({error}, {ens_err:.3f})', 
                 xy=(error, ens_err), 
                 xytext=(10, 10), 
                 textcoords='offset points',
                 fontsize=9,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

plt.tight_layout()
plt.show()

# 5. Сравнение для разного количества классификаторов
print("\n📊 5. СРАВНЕНИЕ ДЛЯ РАЗНОГО КОЛИЧЕСТВА КЛАССИФИКАТОРОВ")
print("-" * 40)

n_classifiers_list = [3, 5, 7, 9, 11, 15, 21]
base_error = 0.25

print(f'Базовая ошибка: {base_error}')
print(f'\nАнсамблевая ошибка для разного количества классификаторов:')
for n in n_classifiers_list:
    ens_err = ensemble_error(n_classifier=n, error=base_error)
    improvement = (base_error - ens_err) / base_error * 100
    print(f'  n={n:2d}: {ens_err:.6f} (улучшение: {improvement:.2f}%)')

# Визуализация
plt.figure(figsize=(10, 6))
for n in n_classifiers_list:
    ens_errors_n = [ensemble_error(n_classifier=n, error=error) for error in error_range]
    plt.plot(error_range, ens_errors_n, label=f'n={n}', linewidth=2)

plt.plot(error_range, error_range, linestyle='--', label='Базовая ошибка', 
         linewidth=2, color='black', alpha=0.5)
plt.xlabel('Базовая ошибка', fontsize=12)
plt.ylabel('Ансамблевая ошибка', fontsize=12)
plt.title(f'Зависимость ансамблевой ошибки от количества классификаторов', 
          fontsize=14, fontweight='bold')
plt.legend(loc='upper left', fontsize=10)
plt.grid(alpha=0.5)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.tight_layout()
plt.show()

# 6. Анализ порога ошибки = 0.5
print("\n🔍 6. АНАЛИЗ ПОРОГА ОШИБКИ = 0.5")
print("-" * 40)

print('Ключевой инсайт:')
print('  - Если базовая ошибка < 0.5: ансамбль улучшает результат')
print('  - Если базовая ошибка = 0.5: ансамбль не даёт улучшения')
print('  - Если базовая ошибка > 0.5: ансамбль ухудшает результат')

print(f'\nДетальный анализ для n={n_classifier}:')
for error in [0.3, 0.4, 0.45, 0.49, 0.5, 0.51, 0.55, 0.6]:
    ens_err = ensemble_error(n_classifier=n_classifier, error=error)
    diff = error - ens_err
    status = "✅ Улучшение" if diff > 0 else "❌ Ухудшение" if diff < 0 else "⚠️  Без изменений"
    print(f'  E={error:.2f} → Ens={ens_err:.4f} (diff={diff:+.4f}) {status}')

# 7. Практические соображения
print("\n💡 7. ПРАКТИЧЕСКИЕ СООБРАЖЕНИЯ")
print("-" * 40)

print('Предположения теории:')
print('  1. Базовые классификаторы работают лучше случайного угадывания (E < 0.5)')
print('  2. Ошибки классификаторов независимы (некоррелированы)')
print('  3. Все классификаторы имеют одинаковую базовую ошибку')

print('\nВ реальности:')
print('  ⚠️  Ошибки часто коррелированы (обучаются на одних данных)')
print('  ⚠️  Базовые ошибки могут различаться')
print('  ⚠️  Некоторым классификаторам может быть E > 0.5')

print('\nПопулярные ансамблевые методы:')
print('  ✅ Majority Voting (голосование большинством)')
print('  ✅ Bagging (Bootstrap Aggregating)')
print('  ✅ Random Forest')
print('  ✅ Boosting (AdaBoost, Gradient Boosting)')
print('  ✅ Stacking (мета-классификатор на базовых классификаторах)')

# 8. Демонстрация Majority Voting
print("\n🗳️ 8. ДЕМОНСТРАЦИЯ MAJORITY VOTING")
print("-" * 40)

# Пример: 11 классификаторов, каждый с ошибкой 0.25
n_classifiers = 11
base_error = 0.25

print(f'Сценарий: {n_classifiers} классификаторов, каждый с ошибкой {base_error}')
print(f'\nАнсамбль принимает правильное решение, если >= {math.ceil(n_classifiers/2.)} классификаторов правы')
print(f'Ансамбль ошибается, если >= {math.ceil(n_classifiers/2.)} классификаторов ошибаются')

# Симуляция одного примера
print(f'\nСимуляция одного примера (majority voting):')
np.random.seed(42)
predictions = np.random.rand(n_classifiers) > base_error  # True = правильное предсказание
correct_votes = np.sum(predictions)
wrong_votes = n_classifiers - correct_votes

print(f'  Правильных голосов: {correct_votes}')
print(f'  Неправильных голосов: {wrong_votes}')
if correct_votes > wrong_votes:
    print(f'  ✅ Ансамбль принимает ПРАВИЛЬНОЕ решение')
else:
    print(f'  ❌ Ансамбль принимает НЕПРАВИЛЬНОЕ решение')

# Множественная симуляция для оценки
print(f'\nМножественная симуляция (10000 примеров):')
n_simulations = 10000
ensemble_correct = 0
for _ in range(n_simulations):
    predictions = np.random.rand(n_classifiers) > base_error
    if np.sum(predictions) > n_classifiers / 2:
        ensemble_correct += 1

ensemble_accuracy = ensemble_correct / n_simulations
ensemble_error_sim = 1 - ensemble_accuracy
theoretical_error = ensemble_error(n_classifier=n_classifiers, error=base_error)

print(f'  Симулированная точность ансамбля: {ensemble_accuracy:.4f}')
print(f'  Симулированная ошибка ансамбля: {ensemble_error_sim:.4f}')
print(f'  Теоретическая ошибка ансамбля: {theoretical_error:.4f}')
print(f'  Разница: {abs(ensemble_error_sim - theoretical_error):.6f}')

# 9. Визуализация распределения голосов
print("\n📊 9. ВИЗУАЛИЗАЦИЯ РАСПРЕДЕЛЕНИЯ ГОЛОСОВ")
print("-" * 40)

n_simulations = 10000
vote_distribution = np.zeros(n_classifiers + 1)

for _ in range(n_simulations):
    predictions = np.random.rand(n_classifiers) > base_error
    correct_votes = np.sum(predictions)
    vote_distribution[correct_votes] += 1

vote_distribution = vote_distribution / n_simulations

plt.figure(figsize=(12, 6))
bars = plt.bar(range(n_classifiers + 1), vote_distribution, 
               color=['lightcoral' if i <= n_classifiers/2 else 'lightgreen' 
                      for i in range(n_classifiers + 1)],
               alpha=0.7, edgecolor='black')
plt.xlabel('Количество правильных голосов', fontsize=12)
plt.ylabel('Вероятность', fontsize=12)
plt.title(f'Распределение правильных голосов ({n_classifiers} классификаторов, E={base_error})', 
          fontsize=14, fontweight='bold')
plt.xticks(range(n_classifiers + 1))
plt.grid(True, alpha=0.3, axis='y')

# Добавляем вертикальную линию для majority threshold
threshold = math.ceil(n_classifiers / 2.)
plt.axvline(x=threshold, color='red', linestyle='--', linewidth=2, 
            label=f'Порог majority (>{threshold} голосов)')
plt.legend(fontsize=10)

# Добавляем значения на столбцы
for i, bar in enumerate(bars):
    height = bar.get_height()
    if height > 0.001:
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.3f}', ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

# 10. Выводы
print("\n📝 10. ВЫВОДЫ")
print("=" * 60)
print("Ансамблевые методы позволяют:")
print("  ✅ Снизить ошибку классификации")
print("  ✅ Улучшить обобщающую способность")
print("  ✅ Уменьшить переобучение")
print("  ✅ Повысить стабильность предсказаний")
print("\nКлючевые условия для эффективности ансамблей:")
print("  - Базовые классификаторы должны работать лучше случайного угадывания")
print("  - Ошибки классификаторов должны быть независимы")
print("  - Разнообразие базовых классификаторов")
print("\nТеоретический результат:")
print(f"  - При n={n_classifier} классификаторах с ошибкой {base_error}")
print(f"  - Ансамблевая ошибка: {ensemble_error(n_classifier=n_classifier, error=base_error):.6f}")
print(f"  - Улучшение: {(base_error - ensemble_error(n_classifier=n_classifier, error=base_error)) / base_error * 100:.2f}%")
print("\nПрактическое применение:")
print("  - Random Forest: ансамбль деревьев решений")
print("  - AdaBoost: последовательное обучение с весами")
print("  - Gradient Boosting: оптимизация функции потерь")
print("  - Stacking: мета-классификатор на базовых моделях")
