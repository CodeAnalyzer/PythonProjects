"""
Раздел 6.1.1. Загрузка набора данных по раку молочной железы в Висконсине
Учебный пример из книги "Python Machine Learning"
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

print("=" * 72)
print("Раздел 6.1.1. Загрузка набора данных по раку молочной железы")
print("=" * 72)

# Загрузка набора данных WDBC из локального файла
print("\n=== 1. Загрузка набора данных WDBC ===")
df = pd.read_csv('D:/GITHUB/PythonProjects/perceptron_classifier/wdbc.data', header=None)

print(f"Размер набора данных: {df.shape}")
print("Первые 5 строк набора данных:")
print(df.head())
print()

# Проверка уникальных диагнозов
print("Уникальные диагнозы (столбец 1):")
print(df[1].unique())
print()

# Разделение на признаки и метки классов
print("=== 2. Разделение на признаки и метки классов ===")
# Столбцы с 3-го по 32-й содержат 30 признаков (индексы 2:32)
X = df.loc[:, 2:].values
# Столбец 1 содержит диагнозы (M - злокачественный, B - доброкачественный)
y = df.loc[:, 1].values

print(f"Размер массива признаков X: {X.shape}")
print(f"Размер массива меток y: {y.shape}")
print()

# Кодирование меток классов с помощью LabelEncoder
print("=== 3. Кодирование меток классов ===")
le = LabelEncoder()
y = le.fit_transform(y)

print("Закодированные классы:")
print(le.classes_)
print(f"M (злокачественный) -> {le.transform(['M'])[0]}")
print(f"B (доброкачественный) -> {le.transform(['B'])[0]}")
print()

# Проверка сопоставления
print("Проверка сопоставления для фиктивных меток:")
print(f"le.transform(['M', 'B']) = {le.transform(['M', 'B'])}")
print()

# Разделение на обучающие и тестовые наборы
print("=== 4. Разделение на обучающие и тестовые наборы ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=1
)

print(f"Размер обучающего набора X_train: {X_train.shape}")
print(f"Размер тестового набора X_test: {X_test.shape}")
print(f"Размер обучающих меток y_train: {y_train.shape}")
print(f"Размер тестовых меток y_test: {y_test.shape}")
print()

# Проверка стратификации (пропорции классов)
print("=== 5. Проверка стратификации ===")
print("Пропорции классов в исходном наборе:")
print(f"Класс 0 (доброкачественный): {np.bincount(y)[0] / len(y):.4f}")
print(f"Класс 1 (злокачественный): {np.bincount(y)[1] / len(y):.4f}")
print()

print("Пропорции классов в обучающем наборе:")
print(f"Класс 0 (доброкачественный): {np.bincount(y_train)[0] / len(y_train):.4f}")
print(f"Класс 1 (злокачественный): {np.bincount(y_train)[1] / len(y_train):.4f}")
print()

print("Пропорции классов в тестовом наборе:")
print(f"Класс 0 (доброкачественный): {np.bincount(y_test)[0] / len(y_test):.4f}")
print(f"Класс 1 (злокачественный): {np.bincount(y_test)[1] / len(y_test):.4f}")
print()

# Статистика по данным
print("=== 6. Статистика по данным ===")
print(f"Всего образцов: {len(y)}")
print(f"Доброкачественных (B, класс 0): {np.bincount(y)[0]}")
print(f"Злокачественных (M, класс 1): {np.bincount(y)[1]}")
print()

print("Средние значения признаков в обучающем наборе:")
print(np.round(X_train.mean(axis=0), 4))
print()

print("Стандартные отклонения признаков в обучающем наборе:")
print(np.round(X_train.std(axis=0), 4))

print("\n" + "=" * 72)
print("Раздел 6.1.2. Объединение преобразователей и оценивателей в конвейер")
print("=" * 72)

# Создание конвейера (pipeline)
print("\n=== 7. Создание конвейера ===")
print("Конвейер включает:")
print("  1. StandardScaler - стандартизация признаков")
print("  2. PCA(n_components=2) - сжатие до 2 главных компонент")
print("  3. LogisticRegression - классификатор")
print()

pipe_lr = make_pipeline(
    StandardScaler(),
    PCA(n_components=2),
    LogisticRegression()
)

print("Объект конвейера:")
print(pipe_lr)
print()

# Обучение конвейера
print("=== 8. Обучение конвейера ===")
pipe_lr.fit(X_train, y_train)
print("Конвейер обучен")
print()

# Предсказание и оценка точности
print("=== 9. Предсказание и оценка точности ===")
y_pred = pipe_lr.predict(X_test)
test_acc = pipe_lr.score(X_test, y_test)
train_acc = pipe_lr.score(X_train, y_train)

print(f"Точность на обучающих данных: {train_acc:.3f}")
print(f"Точность на тестовых данных: {test_acc:.3f}")
print()

# Доступ к отдельным этапам конвейера
print("=== 10. Доступ к этапам конвейера ===")
print("Этапы конвейера:")
for idx, (name, step) in enumerate(pipe_lr.steps, start=1):
    print(f"  {idx}. {name}: {step}")
print()

# Проверка объясненной дисперсии PCA
print("=== 11. Объясненная дисперсия PCA ===")
pca = pipe_lr.named_steps['pca']
print(f"Объясненная дисперсия по компонентам: {pca.explained_variance_ratio_}")
print(f"Совокупная объясненная дисперсия: {pca.explained_variance_ratio_.sum():.4f}")
print()

print("Преимущества конвейера:")
print("  - Автоматическое применение преобразований к train и test")
print("  - Упрощение кода и предотвращение ошибок")
print("  - Легкая настройка гиперпараметров всех этапов")

# Сравнение разного количества компонент PCA
print("\n" + "=" * 72)
print("Сравнение разного количества компонент PCA")
print("=" * 72)

print("\n=== 12. Поиск оптимального количества компонент ===")
print("Точность конвейера при разном количестве компонент PCA:\n")

for n_components in range(2, 31, 2):
    pipe_lr_test = make_pipeline(
        StandardScaler(),
        PCA(n_components=n_components),
        LogisticRegression()
    )
    pipe_lr_test.fit(X_train, y_train)
    train_acc = pipe_lr_test.score(X_train, y_train)
    test_acc = pipe_lr_test.score(X_test, y_test)
    
    # Получаем объясненную дисперсию
    pca = pipe_lr_test.named_steps['pca']
    explained_var = pca.explained_variance_ratio_.sum()
    
    print(f"n_components={n_components:2d} | "
          f"Объясненная дисперсия: {explained_var:.4f} | "
          f"Train accuracy: {train_acc:.3f} | "
          f"Test accuracy: {test_acc:.3f}")

print("\n=== 13. Оптимальное количество компонент ===")
print("Для сохранения 95% дисперсии:")
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
pca_full = PCA().fit(X_train_std)

cum_var = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.argmax(cum_var >= 0.95) + 1

print(f"Необходимо {n_components_95} компонент для объяснения 95% дисперсии")
print(f"Совокупная дисперсия при {n_components_95} компонентах: {cum_var[n_components_95-1]:.4f}")

# Обучение с оптимальным количеством компонент
pipe_lr_optimal = make_pipeline(
    StandardScaler(),
    PCA(n_components=n_components_95),
    LogisticRegression()
)
pipe_lr_optimal.fit(X_train, y_train)

print(f"\nТочность с {n_components_95} компонентами:")
print(f"Train accuracy: {pipe_lr_optimal.score(X_train, y_train):.3f}")
print(f"Test accuracy: {pipe_lr_optimal.score(X_test, y_test):.3f}")
