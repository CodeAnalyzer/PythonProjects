import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, resolution=0.02):
    """Визуализация областей решений для любого классификатора."""
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

print("XOR: КЛАССИЧЕСКАЯ ПРОБЛЕМА НЕЛИНЕЙНОЙ КЛАССИФИКАЦИИ")
print("=" * 60)

# 1. Создание синтетического набора данных XOR
print("\n1. СОЗДАНИЕ НАБОРА ДАННЫХ XOR")
print("-" * 40)
np.random.seed(1)
X_xor = np.random.randn(200, 2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
y_xor = np.where(y_xor, 1, 0)

print(f'Форма X_xor: {X_xor.shape}')
print(f'Форма y_xor: {y_xor.shape}')
print(f'Классы: {np.unique(y_xor)}')
print(f'Распределение классов: {np.bincount(y_xor)}')

# 2. Визуализация исходных данных
print("\n2. ВИЗУАЛИЗАЦИЯ ИСХОДНЫХ ДАННЫХ")
print("-" * 40)

plt.figure(figsize=(12, 4))

# Исходные данные
plt.subplot(1, 3, 1)
plt.scatter(X_xor[y_xor == 1, 0], X_xor[y_xor == 1, 1],
            c='royalblue', marker='s', label='Класс 1', alpha=0.8)
plt.scatter(X_xor[y_xor == 0, 0], X_xor[y_xor == 0, 1],
            c='tomato', marker='o', label='Класс 0', alpha=0.8)
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.title('XOR: Исходные данные')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

# 3. Линейная SVM (провалится)
print("\n3. ЛИНЕЙНАЯ SVM - ПРОВАЛ")
print("-" * 40)

svm_linear = SVC(kernel='linear', random_state=1, C=1.0)
svm_linear.fit(X_xor, y_xor)

accuracy_linear = svm_linear.score(X_xor, y_xor)
print(f'Точность линейной SVM: {accuracy_linear:.3f} ({accuracy_linear*100:.1f}%)')
print(f'Количество опорных векторов: {svm_linear.n_support_.sum()}')

plt.subplot(1, 3, 2)
plot_decision_regions(X_xor, y_xor, svm_linear)
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.title(f'Линейная SVM\nТочность: {accuracy_linear*100:.1f}%')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

# 4. Ядерная SVM с RBF (радиальная базисная функция)
print("\n4. RBF SVM - УСПЕХ")
print("-" * 40)

svm_rbf = SVC(kernel='rbf', random_state=1, C=1.0, gamma=0.3)
svm_rbf.fit(X_xor, y_xor)

accuracy_rbf = svm_rbf.score(X_xor, y_xor)
print(f'Точность RBF SVM: {accuracy_rbf:.3f} ({accuracy_rbf*100:.1f}%)')
print(f'Количество опорных векторов: {svm_rbf.n_support_.sum()}')
print(f'Параметр gamma: {svm_rbf.gamma}')
print(f'Параметр C: {svm_rbf.C}')

plt.subplot(1, 3, 3)
plot_decision_regions(X_xor, y_xor, svm_rbf)
plt.xlim([-3, 3])
plt.ylim([-3, 3])
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.title(f'RBF SVM (gamma=0.3)\nТочность: {accuracy_rbf*100:.1f}%')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 5. Эксперимент с разными параметрами gamma
print("\n5. ЭКСПЕРИМЕНТ С ПАРАМЕТРОМ GAMMA")
print("-" * 40)

gamma_values = [0.1, 0.3, 1.0, 10.0]

plt.figure(figsize=(15, 4))
for i, gamma in enumerate(gamma_values):
    svm = SVC(kernel='rbf', random_state=1, C=1.0, gamma=gamma)
    svm.fit(X_xor, y_xor)
    accuracy = svm.score(X_xor, y_xor)
    
    plt.subplot(1, 4, i+1)
    plot_decision_regions(X_xor, y_xor, svm)
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.title(f'Gamma = {gamma}\nТочность: {accuracy*100:.1f}%\nОпорных: {svm.n_support_.sum()}')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 6. Эксперимент с разными параметрами C
print("\n6. ЭКСПЕРИМЕНТ С ПАРАМЕТРОМ C")
print("-" * 40)

c_values = [0.1, 1.0, 10.0, 100.0]

plt.figure(figsize=(15, 4))
for i, C in enumerate(c_values):
    svm = SVC(kernel='rbf', random_state=1, C=C, gamma=0.3)
    svm.fit(X_xor, y_xor)
    accuracy = svm.score(X_xor, y_xor)
    
    plt.subplot(1, 4, i+1)
    plot_decision_regions(X_xor, y_xor, svm)
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.title(f'C = {C}\nТочность: {accuracy*100:.1f}%\nОпорных: {svm.n_support_.sum()}')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 7. Теоретическое объяснение
print("\n7. ТЕОРЕТИЧЕСКОЕ ОБЪЯСНЕНИЕ")
print("-" * 40)

print("ПРОБЛЕМА XOR:")
print(" XOR - классический пример линейно неразделимых данных")
print(" Класс 1: (x1 > 0 XOR x2 > 0) - противоположные квадранты")
print(" Класс 0: (x1 <= 0 XOR x2 <= 0) - другие противоположные квадранты")
print(" Линейная граница не может разделить эти классы")

print("\nЯДЕРНЫЙ ТРЮК:")
print(" Идея: спроецировать данные в пространство большей размерности")
print(" В новом пространстве данные могут стать линейно разделимыми")
print(" RBF ядро: K(x, x') = exp(-gamma * ||x - x'||²)")

print("\nПАРАМЕТР GAMMA:")
print(" Определяет 'радиус влияния' одной обучающей точки")
print(" Малое gamma: широкий радиус, гладкая граница")
print(" Большое gamma: узкий радиус, извилистая граница, риск переобучения")

print("\nПАРАМЕТР C:")
print(" Регуляризация, как в линейной SVM")
print(" Малое C: широкий отступ, допускает ошибки")
print(" Большое C: узкий отступ, меньше ошибок обучения")

print("\nRBF ЯДРО:")
print(" Самое популярное ядро для SVM")
print(" Теорема об универсальном приближении")
print(" Работает для любых данных, но требует настройки параметров")

print("\nПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ:")
print(" 1. Начните с gamma = 'scale' (автоматический выбор)")
print(" 2. Используйте GridSearchCV для настройки C и gamma")
print(" 3. Стандартизируйте данные перед RBF SVM")
print(" 4. Следите за переобучением при больших gamma и C")

print("\nСРАВНЕНИЕ ЯДЕР:")
print(" Linear: для линейно разделимых данных")
print(" Polynomial: для сложных нелинейных зависимостей")
print(" RBF (Gaussian): универсальное, лучшее для большинства задач")
print(" Sigmoid: редко используется, похоже на нейронную сеть")

# 8. Анализ результатов
print("\n8. АНАЛИЗ РЕЗУЛЬТАТОВ")
print("-" * 40)

print(f"Линейная SVM: {accuracy_linear*100:.1f}% точность")
print(f"RBF SVM (gamma=0.3): {accuracy_rbf*100:.1f}% точность")
print(f"Улучшение: {(accuracy_rbf - accuracy_linear)*100:.1f}% пунктов")

print(f"\nКлючевой вывод:")
print(f" Ядерная SVM решает проблему, неразрешимую для линейных методов")
print(f" XOR - классический пример, где ядерный трюк очень эффективен")

print("\n" + "="*60)
print("ЯДЕРНАЯ SVM - МОЩНЫЙ ИНСТРУМЕНТ ДЛЯ НЕЛИНЕЙНОЙ КЛАССИФИКАЦИИ")
print("="*60)
