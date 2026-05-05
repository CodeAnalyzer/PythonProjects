# -*- coding: utf-8 -*-
"""
Раздел 7.5.5: Использование XGBoost

Пример демонстрирует:
1. Использование XGBClassifier из библиотеки xgboost
2. Сравнение производительности XGBoost с другими методами
3. Объяснение параметров XGBoost

XGBoost (Extreme Gradient Boosting) - оптимизированная реализация градиентного бустинга:
- Последовательное обучение слабых классификаторов
- Оптимизация вычислений для скорости и эффективности
- Регуляризация для предотвращения переобучения
- Обработка пропущенных значений
- Параллельная обработка
"""
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

print("🚀 ИСПОЛЬЗОВАНИЕ XGBOOST")
print("=" * 70)

# Проверка наличия xgboost
try:
    import xgboost as xgb
    print(f"XGBoost версия: {xgb.__version__}")
except ImportError:
    print("❌ Ошибка: библиотека xgboost не установлена")
    print("Установите её командой: pip install xgboost")
    sys.exit(1)

# 1. Загрузка данных Wine
print("\n📂 1. ЗАГРУЗКА ДАННЫХ WINE")
print("-" * 50)

df_wine = pd.read_csv('wine.data', header=None)

df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                  'Alcalinity of ash', 'Magnesium', 'Total phenols',
                  'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                  'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

print(f"Исходный набор данных: {df_wine.shape[0]} образцов, {df_wine.shape[1]} признаков")
print(f"Классы: {df_wine['Class label'].unique()}")

# Отбрасываем класс 1, оставляем только классы 2 и 3
df_wine = df_wine[df_wine['Class label'] != 1]

print(f"\nПосле удаления класса 1: {df_wine.shape[0]} образцов")
print(f"Оставшиеся классы: {df_wine['Class label'].unique()}")

# 2. Выбор признаков и кодирование меток
print("\n🔧 2. ПОДГОТОВКА ДАННЫХ")
print("-" * 50)

# Выбираем два признака: Alcohol и OD280/OD315 of diluted wines
X = df_wine[['Alcohol', 'OD280/OD315 of diluted wines']].values
y = df_wine['Class label'].values

print(f"Выбранные признаки:")
print(f"  - Alcohol (индекс 0)")
print(f"  - OD280/OD315 of diluted wines (индекс 1)")
print(f"Форма X: {X.shape}")
print(f"Форма y: {y.shape}")

# Кодирование меток классов в двоичный формат (0 и 1)
le = LabelEncoder()
y = le.fit_transform(y)

print(f"\nПосле кодирования:")
print(f"  Метки классов: {np.unique(y)}")
print(f"  Соответствие: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 3. Разделение на обучающие и тестовые наборы
print("\n✂️ 3. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩИЕ И ТЕСТОВЫЕ НАБОРЫ")
print("-" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=1,
    stratify=y
)

print(f"Обучающий набор: {X_train.shape[0]} образцов")
print(f"Тестовый набор: {X_test.shape[0]} образцов")
print(f"Распределение классов в обучающем наборе: {np.bincount(y_train)}")
print(f"Распределение классов в тестовом наборе: {np.bincount(y_test)}")

# 4. Создание и обучение XGBoost
print("\n🔧 4. СОЗДАНИЕ И ОБУЧЕНИЕ XGBOOST")
print("-" * 50)

model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=4,
    random_state=1,
    use_label_encoder=False,
    eval_metric='logloss'  # Добавляем для предотвращения предупреждения
)

print("Параметры XGBoost:")
print("  - n_estimators=1000 (количество деревьев/раундов)")
print("  - learning_rate=0.01 (скорость обучения)")
print("  - max_depth=4 (максимальная глубина деревьев)")
print("  - random_state=1 (фиксация случайности)")
print("  - use_label_encoder=False (отключение кодирования меток)")

print("\nОбучение модели...")
gbm = model.fit(X_train, y_train)
print("Обучение завершено")

# 5. Оценка производительности
print("\n📊 5. ОЦЕНКА ПРОИЗВОДИТЕЛЬНОСТИ")
print("-" * 50)

y_train_pred = gbm.predict(X_train)
y_test_pred = gbm.predict(X_test)

gbm_train = accuracy_score(y_train, y_train_pred)
gbm_test = accuracy_score(y_test, y_test_pred)

print(f'Точность XGBoost при обучении/тестировании: {gbm_train:.3f}/{gbm_test:.3f}')

# 6. Сравнение с другими методами (из предыдущих примеров)
print("\n📈 6. СРАВНЕНИЕ С ДРУГИМИ МЕТОДАМИ")
print("-" * 50)

print("Сравнение точности на тестовом наборе (из предыдущих разделов):")
print(f"  Обрубок дерева (max_depth=1): ~0.875")
print(f"  AdaBoost (500 деревьев): ~0.917")
print(f"  Бэггинг (500 деревьев): ~0.917")
print(f"  XGBoost (1000 деревьев): {gbm_test:.3f}")

if gbm_test >= 0.9:
    print("\n✅ XGBoost показывает высокую производительность")
elif gbm_test >= 0.85:
    print("\n✅ XGBoost показывает хорошую производительность")
else:
    print("\nℹ️  XGBoost показывает умеренную производительность")

# 7. Объяснение параметров XGBoost
print("\n📚 7. ОБЪЯСНЕНИЕ ПАРАМЕТРОВ XGBOOST")
print("-" * 50)

print("Основные параметры:")
print("\n1. n_estimators (количество деревьев):")
print("   - Значение: 1000")
print("   - Описание: количество деревьев решений (раундов бустинга)")
print("   - Рекомендация: 100-1000, зависит от сложности задачи")
print("   - Влияние: больше деревьев = больше времени, но потенциально лучше точность")

print("\n2. learning_rate (скорость обучения):")
print("   - Значение: 0.01")
print("   - Описание: масштабирует вклад каждого дерева")
print("   - Рекомендация: 0.01-0.1")
print("   - Влияние: ниже скорость = больше деревьев нужно, но лучше обобщение")
print("   - Компромисс: низкая скорость обучения требует больше деревьев")

print("\n3. max_depth (глубина деревьев):")
print("   - Значение: 4")
print("   - Описание: максимальная глубина каждого дерева")
print("   - Рекомендация: 2-6 для слабых учеников")
print("   - Влияние: больше глубина = сложнее модель, риск переобучения")

print("\n4. use_label_encoder:")
print("   - Значение: False")
print("   - Описание: отключает автоматическое кодирование меток")
print("   - Причина: XGBoost ожидает метки в формате 0, 1, 2, ...")
print("   - Важно: мы используем LabelEncoder для подготовки данных")

print("\nПочему XGBoost популярен:")
print("  ✅ Оптимизированная реализация (быстрее стандартного градиентного бустинга)")
print("  ✅ Встроенная регуляризация (L1 и L2)")
print("  ✅ Обработка пропущенных значений")
print("  ✅ Параллельная и распределенная обработка")
print("  ✅ Поддержка кросс-валидации")
print("  ✅ Совместимость с API scikit-learn")

# 8. Выводы
print("\n📝 8. ВЫВОДЫ")
print("=" * 70)
print("Результаты демонстрируют:")
print(f"  ✅ XGBoost достиг точности {gbm_train:.3f} на обучении")
print(f"  ✅ XGBoost достиг точности {gbm_test:.3f} на тесте")
print("\nПреимущества XGBoost:")
print("  - Высокая производительность и скорость")
print("  - Встроенная регуляризация предотвращает переобучение")
print("  - Автоматическая обработка пропущенных значений")
print("  - Поддержка различных функций потерь")
print("  - Масштабируемость на больших данных")
print("\nПрактические рекомендации:")
print("  - Начинайте с learning_rate=0.01-0.1")
print("  - Используйте max_depth=3-6 для большинства задач")
print("  - Увеличивайте n_estimators при низкой скорости обучения")
print("  - Используйте early_stopping для автоматического выбора")
print("  - Настраивайте гиперпараметры через GridSearchCV")
print("\nСравнение с другими методами:")
print("  - XGBoost обычно превосходит AdaBoost и бэггинг")
print("  - Но требует больше времени для настройки гиперпараметров")
print("  - Лучше работает на больших наборах данных")
