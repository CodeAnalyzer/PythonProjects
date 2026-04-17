"""
Полное руководство по Kernel SVM и ядерному трюку
===============================================

Этот модуль предоставляет комплексный обзор ядерных методов в SVM,
включая математические основы, практические реализации и
сравнения различных ядерных функций.

Ключевые концепции:
1. Ядерный трюк: Вычисление скалярных произведений в пространстве высокой размерности
2. RBF ядро: Гауссова радиально-базисная функция
3. Нелинейная классификация: Решение XOR и других сложных задач
4. Настройка гиперпараметров: Параметры C и gamma
5. Сравнение ядер: Линейное, Полиномиальное, RBF, Сигмоидное

Автор: Реализация на основе "Машинное обучение с PyTorch и Scikit-Learn"
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler

def main():
    """Основная демонстрационная функция"""
    
    print("=" * 60)
    print("ПОЛНОЕ РУКОВОДСТВО ПО KERNEL SVM")
    print("=" * 60)
    print()
    
    # 1. Математические основы
    print("1. МАТЕМАТИЧЕСКИЕ ОСНОВЫ")
    print("-" * 30)
    explain_kernel_mathematics()
    
    # 2. Демонстрация задачи XOR
    print("\n2. ДЕМОНСТРАЦИЯ ЗАДАЧИ XOR")
    print("-" * 30)
    demonstrate_xor_solution()
    
    # 3. Сравнение ядер
    print("\n3. СРАВНЕНИЕ ЯДЕР")
    print("-" * 30)
    compare_kernels()
    
    # 4. Практические советы
    print("\n4. ПРАКТИЧЕСКИЕ СОВЕТЫ")
    print("-" * 30)
    provide_practical_tips()

def explain_kernel_mathematics():
    """Объяснение математических основ ядерного трюка"""
    
    print("Формула ядерного трюка:")
    print("K(x^(i), x^(j)) = <phi(x^(i)), phi(x^(j))>")
    print()
    
    print("Где:")
    print("- phi(x): Функция отображения в пространство высокой размерности")
    print("- <.,.>: Скалярное произведение")
    print("- K(x, y): Ядерная функция")
    print()
    
    print("RBF ядро:")
    print("K(x, y) = exp(-gamma * ||x - y||²)")
    print()
    
    print("Ключевая идея:")
    print("Мы можем вычислить K(x, y) без явного вычисления phi(x)")
    print("Это избегает 'проклятия размерности'!")
    print()

def demonstrate_xor_solution():
    """Демонстрация решения задачи XOR с RBF ядром"""
    
    # Создание XOR данных
    np.random.seed(1)
    X_xor = np.random.randn(200, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, 0)
    
    print(f"Набор данных XOR: {X_xor.shape[0]} образцов, {X_xor.shape[1]} признаков")
    print(f"Распределение классов: {np.bincount(y_xor)}")
    print()
    
    # Сравнение линейного и RBF
    svm_linear = SVC(kernel='linear', random_state=1, C=10.0)
    svm_linear.fit(X_xor, y_xor)
    acc_linear = svm_linear.score(X_xor, y_xor)
    
    svm_rbf = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
    svm_rbf.fit(X_xor, y_xor)
    acc_rbf = svm_rbf.score(X_xor, y_xor)
    
    print("Сравнение производительности:")
    print(f"Линейный SVM:  {acc_linear:.3f} точность")
    print(f"RBF SVM:       {acc_rbf:.3f} точность")
    print(f"Улучшение:     {acc_rbf - acc_linear:.3f}")
    print()
    
    print("Опорные векторы:")
    print(f"Линейный SVM:  {len(svm_linear.support_vectors_)} опорных векторов")
    print(f"RBF SVM:       {len(svm_rbf.support_vectors_)} опорных векторов")
    print()
    
    # Создание визуализации
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plot_decision_regions(X_xor, y_xor, svm_linear)
    plt.title('Линейный SVM (недостаточно для XOR)')
    
    plt.subplot(1, 2, 2)
    plot_decision_regions(X_xor, y_xor, svm_rbf)
    plt.title('RBF SVM (идеально для XOR)')
    
    plt.tight_layout()
    plt.show()

def compare_kernels():
    """Сравнение различных ядерных функций"""
    
    # Создание тестовых наборов данных
    datasets = {
        'Linear': make_classification(n_samples=200, n_features=2, 
                                    n_redundant=0, n_informative=2,
                                    random_state=1),
        'Moons': make_moons(n_samples=200, noise=0.2, random_state=1),
        'Circles': make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=1)
    }
    
    kernels = {
        'Linear': {'kernel': 'linear'},
        'Polynomial': {'kernel': 'poly', 'degree': 3},
        'RBF': {'kernel': 'rbf', 'gamma': 0.1},
        'Sigmoid': {'kernel': 'sigmoid', 'gamma': 0.1}
    }
    
    results = {}
    
    for dataset_name, (X, y) in datasets.items():
        print(f"\n{dataset_name} Dataset:")
        print("-" * 20)
        
        # Стандартизация
        sc = StandardScaler()
        X_std = sc.fit_transform(X)
        
        results[dataset_name] = {}
        
        for kernel_name, params in kernels.items():
            svm = SVC(random_state=1, **params)
            svm.fit(X_std, y)
            accuracy = svm.score(X_std, y)
            n_sv = len(svm.support_vectors_)
            
            results[dataset_name][kernel_name] = {'accuracy': accuracy, 'n_sv': n_sv}
            print(f"{kernel_name:12}: Acc={accuracy:.3f}, SV={n_sv}")
    
    # Сводка
    print("\nСВОДКА:")
    print("-" * 30)
    print("Лучшее ядро для каждого набора данных:")
    for dataset_name in datasets:
        best_kernel = max(results[dataset_name].items(), 
                         key=lambda x: x[1]['accuracy'])
        print(f"{dataset_name:8}: {best_kernel[0]} ({best_kernel[1]['accuracy']:.3f})")

def plot_decision_regions(X, y, classifier):
    """Построение областей принятия решений (упрощенная версия)"""
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # Поверхность принятия решений
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                           np.arange(x2_min, x2_max, 0.02))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # Построение образцов
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[idx], marker=markers[idx],
                    label=f'Class {cl}', edgecolor='black')

def provide_practical_tips():
    """Предоставление практических советов по использованию kernel SVM"""
    
    print("Руководство по выбору ядра:")
    print("-" * 25)
    print("1. Линейное ядро:")
    print("   - Использовать для: Данных высокой размерности, больших наборов данных")
    print("   - Преимущества: Быстрое, простое, меньше склонно к переобучению")
    print("   - Параметры: C (регуляризация)")
    print()
    
    print("2. RBF ядро:")
    print("   - Использовать для: Большинства случаев, нелинейных задач")
    print("   - Преимущества: Универсальное, бесконечномерное пространство")
    print("   - Параметры: C, gamma")
    print("   - Совет: Начать с gamma=1/n_features")
    print()
    
    print("3. Полиномиальное ядро:")
    print("   - Использовать для: Когда важны взаимодействия признаков")
    print("   - Преимущества: Интерпретируемый параметр степени")
    print("   - Параметры: C, degree, gamma, coef0")
    print()
    
    print("4. Сигмоидное ядро:")
    print("   - Использовать для: Поведения, подобного нейронной сети")
    print("   - Преимущества: Похоже на нейронные сети")
    print("   - Параметры: C, gamma, coef0")
    print()
    
    print("Настройка гиперпараметров:")
    print("-" * 25)
    print("Стратегия поиска по сетке:")
    print("C: [0.1, 1, 10, 100]")
    print("gamma: [0.01, 0.1, 1, 10]")
    print()
    
    print("Эмпирическое правило:")
    print("- Если недообучение: Увеличить C, увеличить gamma")
    print("- Если переобучение: Уменьшить C, уменьшить gamma")
    print("- Если слишком медленно: Использовать линейное ядро или уменьшить C")
    print()
    
    print("Вычислительные соображения:")
    print("-" * 30)
    print("Сложность: O(n² * n_features) до O(n³ * n_features)")
    print("Память: O(n²) для ядерной матрицы")
    print("Для n > 10,000: Рассмотреть LinearSVC или методы аппроксимации")

if __name__ == "__main__":
    main()
