import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC

def plot_decision_regions(X, y, classifier, resolution=0.02):
    """ визуализация областей принятия решений """
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # построение поверхности принятия решений
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    
    # построение образцов классов
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=cl,
                    edgecolor='black')

def create_xor_data():
    """Создание набора данных XOR"""
    np.random.seed(1)
    X_xor = np.random.randn(200, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0,
                           X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, 0)
    return X_xor, y_xor

def demonstrate_kernel_trick():
    """Демонстрация ядерного трюка с RBF SVM"""
    
    # Create XOR data
    X_xor, y_xor = create_xor_data()
    
    print("=== SVM с RBF ядром для задачи XOR ===")
    print(f"Форма набора данных XOR: {X_xor.shape}")
    print(f"Распределение классов: {np.bincount(y_xor)}")
    print()
    
    # Create and train RBF SVM
    svm = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0)
    svm.fit(X_xor, y_xor)
    
    print("Параметры RBF SVM:")
    print(f"  Ядро: {svm.kernel}")
    print(f"  Гамма: {svm.gamma}")
    print(f"  C: {svm.C}")
    print(f"  Количество опорных векторов: {svm.n_support_}")
    print()
    
    # Calculate accuracy
    accuracy = svm.score(X_xor, y_xor)
    print(f"Точность обучения: {accuracy:.3f}")
    print()
    
    # Построение областей принятия решений
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plot_decision_regions(X_xor, y_xor, classifier=svm)
    plt.title('Граница решений RBF SVM\n(gamma=0.10, C=10.0)')
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.legend(loc='upper left')
    
    # Сравнение с линейным SVM
    plt.subplot(1, 2, 2)
    svm_linear = SVC(kernel='linear', random_state=1, C=10.0)
    svm_linear.fit(X_xor, y_xor)
    plot_decision_regions(X_xor, y_xor, classifier=svm_linear)
    plt.title('Граница решений линейного SVM\n(C=10.0)')
    plt.xlabel('Признак 1')
    plt.ylabel('Признак 2')
    plt.legend(loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Демонстрация различных значений гамма
    plt.figure(figsize=(15, 5))
    
    gamma_values = [0.01, 0.10, 1.0, 10.0]
    
    for i, gamma in enumerate(gamma_values, 1):
        plt.subplot(1, 4, i)
        svm_gamma = SVC(kernel='rbf', random_state=1, gamma=gamma, C=10.0)
        svm_gamma.fit(X_xor, y_xor)
        
        plot_decision_regions(X_xor, y_xor, classifier=svm_gamma)
        plt.title(f'RBF SVM (gamma={gamma})')
        plt.xlabel('Признак 1')
        if i == 1:
            plt.ylabel('Признак 2')
        
        accuracy = svm_gamma.score(X_xor, y_xor)
        plt.text(0.02, 0.98, f'Acc: {accuracy:.2f}', 
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def explain_kernel_trick():
    """Объяснение концепции ядерного трюка"""
    print("=== Объяснение ядерного трюка ===")
    print()
    print("1. Проблема: Нелинейные данные не могут быть разделены линейной гиперплоскостью")
    print("2. Решение: Отобразить данные в пространство более высокой размерности, где они становятся линейно разделимыми")
    print("3. Сложность: Явное отображение вычислительно дорого")
    print("4. Ядерный трюк: Использовать ядерную функцию для вычисления скалярных произведений в пространстве высокой размерности")
    print()
    print("Формула RBF ядра:")
    print("K(x^(i), x^(j)) = exp(-gamma * ||x^(i) - x^(j)||²)")
    print()
    print("Интерпретация:")
    print("- Измеряет сходство между двумя точками")
    print("- Диапазон: [0, 1], где 1 означает идентичные, 0 означает очень разные")
    print("- Параметр gamma контролирует 'радиус влияния' каждого обучающего примера")
    print()
    print("Влияние гамма:")
    print("- Маленькая гамма: большой радиус сходства, более гладкая граница решений")
    print("- Большая гамма: маленький радиус сходства, более сложная граница")
    print()

def demonstrate_rbf_kernel_computation():
    """Демонстрация вычисления RBF ядра"""
    print("=== Пример вычисления RBF ядра ===")
    print()
    
    # Создание простых примеров точек
    x1 = np.array([1.0, 2.0])
    x2 = np.array([1.1, 2.1])
    x3 = np.array([5.0, 6.0])
    
    gamma = 0.1
    
    def rbf_kernel(x1, x2, gamma):
        """Вычисление RBF ядра между двумя точками"""
        distance_squared = np.sum((x1 - x2) ** 2)
        return np.exp(-gamma * distance_squared)
    
    print(f"Точка 1: {x1}")
    print(f"Точка 2: {x2}")
    print(f"Точка 3: {x3}")
    print(f"Гамма: {gamma}")
    print()
    
    # Вычисление ядер
    k12 = rbf_kernel(x1, x2, gamma)
    k13 = rbf_kernel(x1, x3, gamma)
    k23 = rbf_kernel(x2, x3, gamma)
    
    print("Значения RBF ядра:")
    print(f"K(x1, x2) = {k12:.6f} (похожие точки)")
    print(f"K(x1, x3) = {k13:.6f} (разные точки)")
    print(f"K(x2, x3) = {k23:.6f} (разные точки)")
    print()
    
    # Показать вычисление расстояния
    dist12 = np.sqrt(np.sum((x1 - x2) ** 2))
    dist13 = np.sqrt(np.sum((x1 - x3) ** 2))
    
    print("Евклидовы расстояния:")
    print(f"||x1 - x2|| = {dist12:.6f}")
    print(f"||x1 - x3|| = {dist13:.6f}")
    print()
    print("Заметьте: Меньшее расстояние -> Большее значение ядра (больше сходства)")

if __name__ == "__main__":
    # Запуск демонстраций
    explain_kernel_trick()
    demonstrate_rbf_kernel_computation()
    print("\n" + "="*50 + "\n")
    demonstrate_kernel_trick()
