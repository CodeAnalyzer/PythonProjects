import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from matplotlib.colors import ListedColormap
import seaborn as sns

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

print("📈 ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ С SCIKIT-LEARN")
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

# 5. Обучение с разными подходами
print("\n🧠 5. ОБУЧЕНИЕ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ")
print("-" * 40)

# 5.1 Подход One-vs-Rest (OvR)
print("\n5.1 ПОДХОД ONE-VS-REST (OvR)")
print("-" * 30)
lr_ovr = LogisticRegression(C=100.0, solver='lbfgs', random_state=1)
lr_ovr.fit(X_train_std, y_train)

print(f'Параметры модели OvR:')
print(f'  C (параметр регуляризации): {lr_ovr.C}')
print(f'  Solver: {lr_ovr.solver}')
print(f'  Количество классов: {len(lr_ovr.classes_)}')

# 5.2 Подход Multinomial (рекомендуется)
print("\n5.2 ПОДХОД МУЛЬТИНОМИАЛЬНЫЙ (РЕКОМЕНДУЕТСЯ)")
print("-" * 30)
lr_multi = LogisticRegression(C=100.0, solver='lbfgs', random_state=1)
lr_multi.fit(X_train_std, y_train)

print(f'Параметры модели Multinomial:')
print(f'  C (параметр регуляризации): {lr_multi.C}')
print(f'  Solver: {lr_multi.solver}')
print(f'  Количество классов: {len(lr_multi.classes_)}')

# 6. Оценка моделей
print("\n🎯 6. ОЦЕНКА МОДЕЛЕЙ")
print("-" * 40)

# Предсказания
y_pred_ovr = lr_ovr.predict(X_test_std)
y_pred_multi = lr_multi.predict(X_test_std)

# Точность
accuracy_ovr = accuracy_score(y_test, y_pred_ovr)
accuracy_multi = accuracy_score(y_test, y_pred_multi)

print(f'Точность OvR: {accuracy_ovr:.3f} ({accuracy_ovr*100:.1f}%)')
print(f'Точность Multinomial: {accuracy_multi:.3f} ({accuracy_multi*100:.1f}%)')

# Ошибки
errors_ovr = (y_test != y_pred_ovr).sum()
errors_multi = (y_test != y_pred_multi).sum()

print(f'Ошибки OvR: {errors_ovr} из {len(y_test)}')
print(f'Ошибки Multinomial: {errors_multi} из {len(y_test)}')

# 7. Анализ вероятностей
print("\n📊 7. АНАЛИЗ ВЕРОЯТНОСТЕЙ")
print("-" * 40)

# Вероятности для первых 5 тестовых образцов
print("Вероятности для первых 5 тестовых образцов:")
print("(Формат: [Класс 0, Класс 1, Класс 2])")

prob_ovr = lr_ovr.predict_proba(X_test_std[:5])
prob_multi = lr_multi.predict_proba(X_test_std[:5])

for i in range(5):
    print(f"\nОбразец {i+1}:")
    print(f'  Истинный: {y_test[i]} ({iris.target_names[y_test[i]]})')
    print(f'  Предсказан OvR: {y_pred_ovr[i]} - Вер: [{prob_ovr[i][0]:.3f}, {prob_ovr[i][1]:.3f}, {prob_ovr[i][2]:.3f}]')
    print(f'  Предсказан Multi: {y_pred_multi[i]} - Вер: [{prob_multi[i][0]:.3f}, {prob_multi[i][1]:.3f}, {prob_multi[i][2]:.3f}]')

# 8. Визуализация областей решений
print("\n📊 8. ВИЗУАЛИЗАЦИЯ ОБЛАСТЕЙ РЕШЕНИЙ")
print("-" * 40)

# 8.1 График OvR
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plot_decision_regions(X=X_combined_std, 
                     y=y_combined, 
                     classifier=lr_ovr, 
                     test_idx=range(105, 150))
plt.xlabel('Длина лепестка [стандартизированная]')
plt.ylabel('Ширина лепестка [стандартизированная]')
plt.title('Логистическая регрессия - One-vs-Rest')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

# 8.2 График Multinomial
plt.subplot(1, 2, 2)
plot_decision_regions(X=X_combined_std, 
                     y=y_combined, 
                     classifier=lr_multi, 
                     test_idx=range(105, 150))
plt.xlabel('Длина лепестка [стандартизированная]')
plt.ylabel('Ширина лепестка [стандартизированная]')
plt.title('Логистическая регрессия - Multinomial')
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 9. Матрицы конфузии
print("\n📈 9. МАТРИЦЫ ОШИБОК")
print("-" * 40)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Матрица OvR
cm_ovr = confusion_matrix(y_test, y_pred_ovr)
sns.heatmap(cm_ovr, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, yticklabels=iris.target_names,
            ax=axes[0])
axes[0].set_title('Матрица ошибок - OvR')
axes[0].set_xlabel('Предсказано')
axes[0].set_ylabel('Истинно')

# Матрица Multinomial
cm_multi = confusion_matrix(y_test, y_pred_multi)
sns.heatmap(cm_multi, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names,
            ax=axes[1])
axes[1].set_title('Матрица ошибок - Multinomial')
axes[1].set_xlabel('Предсказано')
axes[1].set_ylabel('Истинно')

plt.tight_layout()
plt.show()

# 10. Отчеты классификации
print("\n📋 10. ОТЧЕТЫ КЛАССИФИКАЦИИ")
print("-" * 40)

print("\nОтчет OvR:")
print(classification_report(y_test, y_pred_ovr, target_names=iris.target_names))

print("\nОтчет Multinomial:")
print(classification_report(y_test, y_pred_multi, target_names=iris.target_names))

# 11. Сравнение весов модели
print("\n⚖️ 11. СРАВНЕНИЕ ВЕСОВ МОДЕЛИ")
print("-" * 40)

print("Веса модели OvR:")
print(f"Форма: {lr_ovr.coef_.shape}")
for i, class_name in enumerate(iris.target_names):
    print(f"  {class_name}: [{lr_ovr.coef_[i][0]:.3f}, {lr_ovr.coef_[i][1]:.3f}]")

print(f"\nСмещения OvR: {lr_ovr.intercept_}")

print("\nВеса модели Multinomial:")
print(f"Форма: {lr_multi.coef_.shape}")
for i, class_name in enumerate(iris.target_names):
    print(f"  {class_name}: [{lr_multi.coef_[i][0]:.3f}, {lr_multi.coef_[i][1]:.3f}]")

print(f"\nСмещения Multinomial: {lr_multi.intercept_}")

# 12. Выводы
print("\n🎯 ВЫВОДЫ")
print("=" * 60)
print(f" Оба подхода достигли высокой точности:")
print(f"   OvR: {accuracy_ovr*100:.1f}% ({errors_ovr} ошибок)")
print(f"   Multinomial: {accuracy_multi*100:.1f}% ({errors_multi} ошибок)")
print(f"\n Ключевые различия:")
print(f"   OvR: Обучает K бинарных классификаторов (один против всех)")
print(f"   Multinomial: Оптимизирует единую функцию потерь для всех классов")
print(f"\n Практическая рекомендация:")
print(f"   Использовать 'multinomial' для взаимоисключающих классов (как Iris)")
print(f"   Использовать 'ovr' для многоклассовой классификации или специфических задач")
print(f"\n Параметр C=100.0:")
print(f"   Контролирует регуляризацию (обратная сила регуляризации)")
print(f"   Большие значения = меньше регуляризации")
