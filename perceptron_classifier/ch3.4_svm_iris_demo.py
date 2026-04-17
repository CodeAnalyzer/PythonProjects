import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    """Визуализация областей решений с выделением тестовых образцов."""
    # Настройка генератора маркеров и цветовой карты
    markers = ('s', 'x', 'o', '^', 'v')
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
                    label=f'Класс {cl}', 
                    edgecolor='black')
    
    # Выделение тестовых образцов
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    c='none', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='Тестовый набор')

print("🎯 SVM КЛАССИФИКАЦИЯ НАБОР ДАННЫХ IRIS")
print("=" * 60)

# 1. Загрузка набора данных Iris
print("\n📊 1. ЗАГРУЗКА ДАННЫХ IRIS")
print("-" * 40)
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]  # Длина и ширина лепестков
y = iris.target

print(f'Классы: {iris.target_names}')
print(f'Признаки: {iris.feature_names[2:4]}')
print(f'Форма X: {X.shape}')
print(f'Форма y: {y.shape}')
print(f'Метки классов: {np.unique(y)}')

# Проверка распределения классов
print(f'\nРаспределение классов:')
for i, class_name in enumerate(iris.target_names):
    count = np.sum(y == i)
    print(f'  {class_name}: {count} образцов')

# 2. Разделение на обучающие и тестовые наборы
print("\n🔀 2. РАЗДЕЛЕНИЕ ДАННЫХ")
print("-" * 40)
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
print("-" * 40)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

print(f'Средние значения (обучающие): [{X_train.mean(axis=0)[0]:.2f}, {X_train.mean(axis=0)[1]:.2f}]')
print(f'Стандартные отклонения (обучающие): [{X_train.std(axis=0)[0]:.2f}, {X_train.std(axis=0)[1]:.2f}]')
print(f'Средние значения (стандартизированные): [{X_train_std.mean(axis=0)[0]:.6f}, {X_train_std.mean(axis=0)[1]:.6f}]')
print(f'Стандартные отклонения (стандартизированные): [{X_train_std.std(axis=0)[0]:.6f}, {X_train_std.std(axis=0)[1]:.6f}]')

# 4. Объединение данных для визуализации
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# 5. Обучение SVM с линейным ядром
print("\n🧠 5. ОБУЧЕНИЕ SVM С ЛИНЕЙНЫМ ЯДРОМ")
print("-" * 40)
svm = SVC(kernel='linear', C=1.0, random_state=1)
svm.fit(X_train_std, y_train)

print(f'Параметры модели SVM:')
print(f'  Kernel: {svm.kernel}')
print(f'  C (параметр регуляризации): {svm.C}')
print(f'  Количество классов: {len(svm.classes_)}')
print(f'  Количество опорных векторов: {svm.n_support_.sum()}')
print(f'  Опорные векторы по классам: {svm.n_support_}')

# 6. Оценка модели
print("\n🎯 6. ОЦЕНКА МОДЕЛИ")
print("-" * 40)

# Предсказания
y_pred = svm.predict(X_test_std)

# Точность
accuracy = svm.score(X_test_std, y_test)
print(f'Точность на тестовом наборе: {accuracy:.3f} ({accuracy*100:.1f}%)')

# Точность на обучающем наборе
train_accuracy = svm.score(X_train_std, y_train)
print(f'Точность на обучающем наборе: {train_accuracy:.3f} ({train_accuracy*100:.1f}%)')

# Ошибки
errors = (y_test != y_pred).sum()
print(f'Ошибки: {errors} из {len(y_test)}')

# 7. Анализ опорных векторов
print("\n📊 7. АНАЛИЗ ОПОРНЫХ ВЕКТОРОВ")
print("-" * 40)

print(f'Общее количество опорных векторов: {svm.n_support_.sum()}')
for i, class_name in enumerate(iris.target_names):
    print(f'  {class_name}: {svm.n_support_[i]} опорных векторов')

# Коэффициенты модели (для линейного ядра)
if svm.kernel == 'linear':
    print(f'\nКоэффициенты модели:')
    print(f'Форма: {svm.coef_.shape}')
    for i, class_name in enumerate(iris.target_names):
        print(f'  {class_name}: [{svm.coef_[i][0]:.3f}, {svm.coef_[i][1]:.3f}]')
    print(f'Смещения: {svm.intercept_}')

# 8. Визуализация областей решений
print("\n📊 8. ВИЗУАЛИЗАЦИЯ ОБЛАСТЕЙ РЕШЕНИЙ")
print("-" * 40)

plt.figure(figsize=(10, 6))
plot_decision_regions(X=X_combined_std, 
                     y=y_combined, 
                     classifier=svm, 
                     test_idx=range(105, 150))
plt.xlabel('Длина лепестка [стандартизированная]')
plt.ylabel('Ширина лепестка [стандартизированная]')
plt.title('SVM с линейным ядром - Классификация Iris')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 9. Теоретическое объяснение
print("\n📚 9. ТЕОРЕТИЧЕСКОЕ ОБЪЯСНЕНИЕ")
print("-" * 40)

print("SUPPORT VECTOR MACHINE (SVM)")
print("\nОСНОВНАЯ ИДЕЯ:")
print("• SVM ищет оптимальную разделяющую гиперплоскость")
print("• Максимизирует отступ (margin) между классами")
print("• Опорные векторы - образцы на границе отступа")

print("\nЛИНЕЙНОЕ ЯДРО:")
print("• Используется для линейно разделимых данных")
print("• Функция решения: f(x) = w^T x + b")
print("• Классификация: sign(f(x))")

print("\nПАРАМЕТР C:")
print("• Контролирует баланс между шириной отступа и ошибками классификации")
print("• Большое C: узкий отступ, меньше ошибок обучения (риск переобучения)")
print("• Малое C: широкий отступ, больше ошибок обучения (лучше обобщение)")

print("\nОПОРНЫЕ ВЕКТОРЫ:")
print("• Критические образцы, определяющие границу решения")
print("• Лежат на границе отступа или внутри него")
print("• Только они влияют на положение разделяющей гиперплоскости")

print("\nПРЕИМУЩЕСТВА SVM:")
print("• Эффективна в высокоразмерных пространствах")
print("• Хорошо работает с малыми наборами данных")
print("• Устойчива к переобучению при правильном выборе C")

print("\n🎯 ВЫВОДЫ")
print("=" * 60)
print(f" SVM с линейным ядром достиг точности {accuracy*100:.1f}% на тестовом наборе")
print(f" Модель использует {svm.n_support_.sum()} опорных векторов")
print(f" Линейное ядро подходит для Iris, так как классы хорошо разделимы")
print(f" Параметр C=1.0 обеспечивает хороший баланс между отступом и ошибками")
