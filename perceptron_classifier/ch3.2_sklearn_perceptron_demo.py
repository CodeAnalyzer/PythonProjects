import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    """Визуализация областей решений с выделением тестовых образцов."""
    # Настройка генератора маркеров и цветовой карты
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # Построение поверхности решений
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    
    lab = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    lab = lab.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, lab, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # Отображение образцов классов
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=f'Class {cl}', 
                    edgecolor='black')
    
    # Выделение тестовых образцов
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='none', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='Test set')

print("🤖 ДЕМО: PERCEPTRON ИЗ SCIKIT-LEARN")
print("=" * 50)

# 1. Загрузка набора данных Iris
print("\n📊 1. ЗАГРУЗКА ДАННЫХ IRIS")
print("-" * 30)
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]  # Длина и ширина лепестков
y = iris.target

print(f'Классы: {iris.target_names}')
print(f'Признаки: {iris.feature_names[2:4]}')
print(f'Форма X: {X.shape}')
print(f'Форма y: {y.shape}')
print(f'Метки классов:', np.unique(y))

# Проверка распределения классов
print(f'\nРаспределение классов:')
for i, class_name in enumerate(iris.target_names):
    count = np.sum(y == i)
    print(f'  {class_name}: {count} образцов')

# 2. Разделение на обучающие и тестовые наборы
print("\n🔀 2. РАЗДЕЛЕНИЕ ДАННЫХ")
print("-" * 30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

print(f'Обучающий набор: {X_train.shape[0]} образцов')
print(f'Тестовый набор: {X_test.shape[0]} образцов')
print(f'Соотношение 70/30 соблюдено')

# Проверка стратификации
print(f'\nСтратификация (сохранение пропорций):')
print(f'  Полный набор: {np.bincount(y)}')
print(f'  Обучающий: {np.bincount(y_train)}')
print(f'  Тестовый: {np.bincount(y_test)}')

# 3. Стандартизация признаков
print("\n⚖️ 3. СТАНДАРТИЗАЦИЯ ПРИЗНАКОВ")
print("-" * 30)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

print(f'Средние значения (обучающие): [{X_train.mean(axis=0)[0]:.2f}, {X_train.mean(axis=0)[1]:.2f}]')
print(f'Стандартные отклонения (обучающие): [{X_train.std(axis=0)[0]:.2f}, {X_train.std(axis=0)[1]:.2f}]')
print(f'Средние значения (стандартизированные): [{X_train_std.mean(axis=0)[0]:.6f}, {X_train_std.mean(axis=0)[1]:.6f}]')
print(f'Стандартные отклонения (стандартизированные): [{X_train_std.std(axis=0)[0]:.6f}, {X_train_std.std(axis=0)[1]:.6f}]')

# 4. Обучение модели Perceptron
print("\n🧠 4. ОБУЧЕНИЕ PERCEPTRON")
print("-" * 30)
ppn = Perceptron(eta0=0.1, random_state=1)
ppn.fit(X_train_std, y_train)

print(f'Параметры модели:')
print(f'  Скорость обучения (eta0): {ppn.eta0}')
print(f'  Количество эпох: {ppn.max_iter}')
print(f'  Случайное состояние: {ppn.random_state}')

# 5. Предсказание и оценка
print("\n🎯 5. ПРЕДСКАЗАНИЕ И ОЦЕНКА")
print("-" * 30)
y_pred = ppn.predict(X_test_std)

# Подсчет ошибок
misclassified = (y_test != y_pred).sum()
error_rate = misclassified / len(y_test)
accuracy = 1 - error_rate

print(f'Ошибочно классифицировано: {misclassified} из {len(y_test)}')
print(f'Ошибка классификации: {error_rate:.3f} ({error_rate*100:.1f}%)')
print(f'Точность классификации: {accuracy:.3f} ({accuracy*100:.1f}%)')

# Альтернативные способы оценки точности
accuracy_sklearn = accuracy_score(y_test, y_pred)
accuracy_method = ppn.score(X_test_std, y_test)

print(f'\nТочность (accuracy_score): {accuracy_sklearn:.3f}')
print(f'Точность (метод score): {accuracy_method:.3f}')

# 6. Анализ ошибок по классам
print("\n📈 6. АНАЛИЗ ОШИБОК ПО КЛАССАМ")
print("-" * 30)
from collections import defaultdict

errors_by_class = defaultdict(int)
total_by_class = defaultdict(int)

for true_label, pred_label in zip(y_test, y_pred):
    total_by_class[true_label] += 1
    if true_label != pred_label:
        errors_by_class[true_label] += 1

print(f'Ошибки по классам:')
for class_id in range(3):
    class_name = iris.target_names[class_id]
    errors = errors_by_class[class_id]
    total = total_by_class[class_id]
    class_accuracy = (total - errors) / total * 100
    print(f'  {class_name}: {errors} ошибок из {total} ({class_accuracy:.1f}% точность)')

# 7. Визуализация областей решений
print("\n📊 7. ВИЗУАЛИЗАЦИЯ ОБЛАСТЕЙ РЕШЕНИЙ")
print("-" * 30)

# Объединение обучающих и тестовых данных для визуализации
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# Создание графика
plt.figure(figsize=(12, 8))
plot_decision_regions(X=X_combined_std, 
                     y=y_combined, 
                     classifier=ppn, 
                     test_idx=range(105, 150))

plt.xlabel('Длина лепестка [стандартизированная]')
plt.ylabel('Ширина лепестка [стандартизированная]')
plt.title('Perceptron из scikit-learn - Области решений')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 8. Сравнение с нашей реализацией
print("\n🔍 8. СРАВНЕНИЕ С НАШЕЙ РЕАЛИЗАЦИЕЙ")
print("-" * 30)

# Загрузка данных как в наших примерах
df = pd.read_csv('iris.data', header=None, encoding='utf-8')
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']

# Преобразование в тот же формат
X_our = df.iloc[:, [2, 3]].values  # petal_length, petal_width
y_our = df.iloc[:, 4].values
y_our_encoded = np.where(y_our == 'Iris-setosa', 0, 
                         np.where(y_our == 'Iris-versicolor', 1, 2))

print(f'Сравнение наборов данных:')
print(f'  scikit-learn: {X.shape} образцов, {len(np.unique(y))} класса')
print(f'  Наша реализация: {X_our.shape} образцов, {len(np.unique(y_our_encoded))} класса')

# Проверка совпадения данных
print(f'\nСовпадение данных:')
print(f'  Признаки совпадают: {np.allclose(X, X_our)}')
print(f'  Метки совпадают: {np.array_equal(y, y_our_encoded)}')

print(f'\n💡 КЛЮЧЕВЫЕ ПРЕИМУЩЕСТВА SCIKIT-LEARN:')
print("✅ Готовые оптимизированные реализации")
print("✅ Поддержка многоклассовой классификации (One-vs-Rest)")
print("✅ Стандартизированный API (fit, predict, score)")
print("✅ Встроенные метрики качества")
print("✅ Интеграция с пайплайнами обработки данных")
print("✅ Множество дополнительных параметров и методов")

print(f'\n🎯 ВЫВОДЫ:')
print(f"• scikit-learn Perceptron достиг {accuracy*100:.1f}% точности")
print(f"• Модель неправильно классифицировала {misclassified} из {len(y_test)} образцов")
print(f"• Три класса не идеально линейно разделимы")
print(f"• API scikit-learn значительно удобнее для практического применения")
