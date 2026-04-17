import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def kernel_trick_mathematical_explanation():
    """Математическое объяснение ядерного трюка"""
    
    print("=== Математические основы ядерного трюка ===")
    print()
    
    print("1. Исходная задача оптимизации SVM:")
    print("   максимизировать: W(alfa) = sum(alfa_i) - 0.5 * sum(alfa_i * alfa_j * y_i * y_j * <x_i, x_j>)")
    print("   при условиях: 0 <= alfa_i <= C, sum(alfa_i * y_i) = 0")
    print()
    
    print("2. С ядерным трюком:")
    print("   максимизировать: W(alfa) = sum(alfa_i) - 0.5 * sum(alfa_i * alfa_j * y_i * y_j * K(x_i, x_j))")
    print("   где K(x_i, x_j) = <phi(x_i), phi(x_j)>")
    print()
    
    print("3. Решающая функция:")
    print("   f(x) = sign(sum(alfa_i * y_i * K(x_i, x) + b))")
    print()
    
    print("4. Свойства RBF ядра:")
    print("   K(x, x') = exp(-gamma * ||x - x'||²)")
    print("   - Положительно определено")
    print("   - Инвариантно к сдвигу")
    print("   - Создает бесконечномерное признаковое пространство")
    print()

def demonstrate_feature_mapping():
    """Демонстрация того, как ядерный трюк избегает явного отображения признаков"""
    
    print("=== Отображение признаков vs Ядерный трюк ===")
    print()
    
    # Простой 2D пример
    X = np.array([[1, 2], [3, 4], [5, 6]])
    
    print("Исходные данные (2D):")
    print(X)
    print()
    
    # Полиномиальное отображение признаков (степень 2)
    print("Явное полиномиальное отображение (степень 2):")
    phi_X = []
    for x in X:
        phi_x = [x[0]**2, np.sqrt(2)*x[0]*x[1], x[1]**2]
        phi_X.append(phi_x)
    phi_X = np.array(phi_X)
    print(phi_X)
    print(f"Новая размерность: {phi_X.shape[1]}")
    print()
    
    # Вычисление скалярного произведения в отображенном пространстве
    dot_product_mapped = np.dot(phi_X[0], phi_X[1])
    print(f"Скалярное произведение в отображенном пространстве: {dot_product_mapped:.4f}")
    print()
    
    # Вычисление с использованием полиномиального ядра
    def polynomial_kernel(x1, x2, degree=2):
        return (np.dot(x1, x2) + 1) ** degree
    
    kernel_value = polynomial_kernel(X[0], X[1])
    print(f"Значение полиномиального ядра: {kernel_value:.4f}")
    print()
    
    print("Заметьте: Ядро дает тот же результат без явного отображения!")
    print()

def visualize_rbf_kernel_surface():
    """Визуализация RBF ядра как поверхности"""
    
    print("=== Визуализация поверхности RBF ядра ===")
    print()
    
    # Создание сетки
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    
    # Центральная точка
    center = np.array([0, 0])
    
    # Вычисление значений RBF ядра
    gamma = 0.5
    Z = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            point = np.array([X[i, j], Y[i, j]])
            distance_squared = np.sum((point - center) ** 2)
            Z[i, j] = np.exp(-gamma * distance_squared)
    
    # Создание визуализации
    fig = plt.figure(figsize=(12, 5))
    
    # 3D график поверхности
    ax1 = fig.add_subplot(121, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis')
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('K(x, center)')
    ax1.set_title(f'Поверхность RBF ядра (gamma={gamma})')
    plt.colorbar(surf, ax=ax1, shrink=0.5)
    
    # 2D контурный график
    ax2 = fig.add_subplot(122)
    contour = ax2.contourf(X, Y, Z, levels=20, cmap='viridis')
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_title('Контур RBF ядра')
    ax2.plot(center[0], center[1], 'r*', markersize=15, label='Центр')
    ax2.legend()
    plt.colorbar(contour, ax=ax2)
    
    plt.tight_layout()
    plt.show()
    
    print("RBF ядро создает 'холм' с центром в каждой обучающей точке.")
    print("Высота и ширина холма контролируются параметром gamma.")

def compare_different_kernels():
    """Сравнение различных ядерных функций"""
    
    print("=== Сравнение различных ядер ===")
    print()
    
    # Тестовые точки
    x1 = np.array([1.0, 1.0])
    x2 = np.array([2.0, 2.0])
    
    # Различные ядра
    def linear_kernel(x1, x2):
        return np.dot(x1, x2)
    
    def polynomial_kernel(x1, x2, degree=2, coef0=1):
        return (np.dot(x1, x2) + coef0) ** degree
    
    def rbf_kernel(x1, x2, gamma=0.5):
        distance_squared = np.sum((x1 - x2) ** 2)
        return np.exp(-gamma * distance_squared)
    
    def sigmoid_kernel(x1, x2, gamma=0.5, coef0=1):
        return np.tanh(gamma * np.dot(x1, x2) + coef0)
    
    print(f"Точка 1: {x1}")
    print(f"Точка 2: {x2}")
    print()
    
    print("Значения ядер:")
    print(f"Linear K(x1, x2)      = {linear_kernel(x1, x2):.4f}")
    print(f"Polynomial K(x1, x2)  = {polynomial_kernel(x1, x2):.4f}")
    print(f"RBF K(x1, x2)         = {rbf_kernel(x1, x2):.4f}")
    print(f"Sigmoid K(x1, x2)     = {sigmoid_kernel(x1, x2):.4f}")
    print()

def demonstrate_gamma_effect():
    """Демонстрация влияния параметра gamma в RBF ядре"""
    
    print("=== Влияние параметра Gamma в RBF ядре ===")
    print()
    
    # Две точки на фиксированном расстоянии
    x1 = np.array([0.0, 0.0])
    x2 = np.array([1.0, 1.0])
    distance = np.sqrt(np.sum((x1 - x2) ** 2))
    
    print(f"Расстояние между точками: {distance:.4f}")
    print()
    
    # Различные значения gamma
    gamma_values = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    
    print("Gamma\t\tЗначение ядра\tИнтерпретация")
    print("-" * 50)
    
    for gamma in gamma_values:
        kernel_value = np.exp(-gamma * distance**2)
        
        if gamma < 0.1:
            interpretation = "Очень гладкая"
        elif gamma < 0.5:
            interpretation = "Гладкая"
        elif gamma < 1.0:
            interpretation = "Умеренная"
        elif gamma < 2.0:
            interpretation = "Сложная"
        else:
            interpretation = "Очень сложная"
        
        print(f"{gamma:.2f}\t\t{kernel_value:.6f}\t{interpretation}")
    
    print()
    print("Высокая gamma -> более локализованное влияние -> более сложная граница решений")
    print("Низкая gamma -> более широкое влияние -> более гладкая граница решений")

if __name__ == "__main__":
    kernel_trick_mathematical_explanation()
    print()
    demonstrate_feature_mapping()
    print()
    compare_different_kernels()
    print()
    demonstrate_gamma_effect()
    print()
    visualize_rbf_kernel_surface()
