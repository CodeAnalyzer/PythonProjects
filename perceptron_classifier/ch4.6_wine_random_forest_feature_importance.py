"""
Раздел 4.6. Оценка важности признаков с помощью случайных лесов

Используя метод случайного леса, мы можем измерить значимость признаков
как усредненное уменьшение примеси, вычисленное по всем деревьям решений
в лесу, не делая никаких предположений о том, являются ли наши данные
линейно разделимыми или нет.

Атрибут feature_importances_ классификатора RandomForestClassifier уже
содержит значения важности признаков.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel


# Загрузка набора данных Wine
print("=" * 60)
print("Раздел 4.6. Оценка важности признаков с помощью случайных лесов")
print("=" * 60)

print("\n=== Загрузка набора данных Wine ===")
df_wine = pd.read_csv('D:/GITHUB/PythonProjects/perceptron_classifier/wine.data', header=None)

df_wine.columns = [
    'Class label', 'Alcohol', 'Malic acid', 'Ash',
    'Alcalinity of ash', 'Magnesium', 'Total phenols',
    'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
    'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline'
]

print("Название признаков:")
for i, col in enumerate(df_wine.columns[1:], 1):
    print(f"  {i:2d}. {col}")

print(f"\nРазмер набора данных: {df_wine.shape[0]} образцов, {df_wine.shape[1]} признаков")
print("Классы:", np.unique(df_wine['Class label']))


# Разделение на признаки и метки классов
print("\n=== Разделение данных ===")
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

# Разделение на обучающие и тестовые наборы
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=0,
    stratify=y
)

feat_labels = df_wine.columns[1:]

print(f"Размер обучающего набора: {X_train.shape}")
print(f"Размер тестового набора: {X_test.shape}")


# Обучение случайного леса
print("\n=== Обучение случайного леса ===")
forest = RandomForestClassifier(
    n_estimators=500,
    random_state=1,
    n_jobs=2  # использовать 2 ядра для параллельного обучения
)

forest.fit(X_train, y_train)

print(f"Количество деревьев в лесу: {forest.n_estimators}")
print(f"Точность на обучающей выборке: {forest.score(X_train, y_train) * 100:.2f}%")
print(f"Точность на тестовой выборке: {forest.score(X_test, y_test) * 100:.2f}%")


# Извлечение важности признаков
print("\n=== Оценка важности признаков ===")
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

print("\nРанжирование признаков по важности:")
print("-" * 60)
for f in range(X_train.shape[1]):
    print("%2d) %-*s %f" % (
        f + 1, 30, feat_labels[indices[f]], importances[indices[f]]
    ))


# Визуализация важности признаков
print("\n=== Построение графика важности признаков ===")
plt.figure(figsize=(10, 8))

# Построение столбчатой диаграммы
plt.bar(
    range(X_train.shape[1]),
    importances[indices],
    align='center',
    color='steelblue'
)

# Настройка осей
plt.xticks(
    range(X_train.shape[1]),
    [feat_labels[i] for i in indices],
    rotation=90,
    fontsize=9,
    ha='right'
)
plt.xlim([-1, X_train.shape[1]])
plt.xlabel('Признаки')
plt.ylabel('Важность признака')
plt.title('Важность признаков (Случайный лес, 500 деревьев)')
plt.grid(axis='y', alpha=0.3)

# Автоматическое масштабирование подписей по оси X
plt.tight_layout()

# Сохранение графика в файл
plt.savefig('d:\\GITHUB\\PythonProjects\\perceptron_classifier\\wine_feature_importance.png', dpi=150)
print("График сохранен как: wine_feature_importance.png")

plt.show()


# Дополнительная информация
print("\n=== Дополнительная информация ===")
print("Сумма важности признаков (должна быть равна 1.0):", sum(importances))
print("\nТоп-5 наиболее важных признаков:")
for i in range(5):
    print(f"  {i+1}. {feat_labels[indices[i]]}: {importances[indices[i]]:.4f}")


# Выбор признаков с помощью SelectFromModel
print("\n" + "=" * 60)
print("Выбор признаков с помощью SelectFromModel")
print("=" * 60)

# Создаем селектор признаков с порогом 0.1
# prefit=True указывает, что модель уже обучена
sfm = SelectFromModel(forest, threshold=0.1, prefit=True)

# Трансформируем данные, выбирая только важные признаки
X_selected = sfm.transform(X_train)

print(f"\nКоличество признаков, соответствующих пороговому критерию: {X_selected.shape[1]}")
print(f"Исходное количество признаков: {X_train.shape[1]}")
print(f"Уменьшение размерности: {X_train.shape[1] - X_selected.shape[1]} признаков удалено")

# Выводим отобранные признаки
print("\nОтобранные признаки (важность > 0.1):")
print("-" * 60)
selected_count = 0
for f in range(X_train.shape[1]):
    if importances[indices[f]] >= 0.1:
        selected_count += 1
        print("%2d) %-*s %f" % (
            selected_count, 30,
            feat_labels[indices[f]],
            importances[indices[f]]
        ))

# Получаем маску выбранных признаков
selected_mask = sfm.get_support()
print("\nМаска выбранных признаков:")
for i, (name, selected) in enumerate(zip(feat_labels, selected_mask), 1):
    status = "✓ ВЫБРАН" if selected else "✗ удален"
    print(f"  {i:2d}. {name:30s} {status}")

# Проверка точности модели с отобранными признаками
print("\n=== Сравнение точности модели ===")

# Разделение тестовых данных
X_test_selected = sfm.transform(X_test)

# Обучение новой модели только на отобранных признаках
forest_selected = RandomForestClassifier(
    n_estimators=500,
    random_state=1,
    n_jobs=2
)
forest_selected.fit(X_selected, y_train)

accuracy_full = forest.score(X_test, y_test)
accuracy_selected = forest_selected.score(X_test_selected, y_test)

print(f"Точность на всех признаках (13):     {accuracy_full * 100:.2f}%")
print(f"Точность на отобранных признаках (5): {accuracy_selected * 100:.2f}%")
print(f"\nРазница: {(accuracy_full - accuracy_selected) * 100:.2f}%")

if accuracy_selected >= accuracy_full:
    print("✓ Модель на отобранных признаках работает не хуже!")
else:
    print("⚠ Небольшое снижение точности при уменьшении количества признаков")
