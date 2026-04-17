import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler

def plot_decision_regions(X, y, classifier, resolution=0.02, title=""):
    """Визуализация областей принятия решений"""
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # Построение поверхности принятия решений
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2_min, xx2.max())
    
    # Построение образцов классов
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
                    y=X[y == cl, 1],
                    alpha=0.8,
                    c=colors[idx],
                    marker=markers[idx],
                    label=f'Class {cl}',
                    edgecolor='black')
    
    plt.title(title)

def create_datasets():
    """Создание различных типов нелинейных наборов данных"""
    
    datasets = {}
    
    # 1. Набор данных moons
    X_moons, y_moons = make_moons(n_samples=200, noise=0.2, random_state=1)
    datasets['moons'] = (X_moons, y_moons)
    
    # 2. Набор данных circles
    X_circles, y_circles = make_circles(n_samples=200, noise=0.2, factor=0.5, random_state=1)
    datasets['circles'] = (X_circles, y_circles)
    
    # 3. XOR-подобный набор данных
    np.random.seed(1)
    X_xor = np.random.randn(200, 2)
    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, 0)
    datasets['xor'] = (X_xor, y_xor)
    
    # 4. Сложная классификация
    X_complex, y_complex = make_classification(n_samples=200, n_features=2, 
                                             n_redundant=0, n_informative=2,
                                             n_clusters_per_class=1, random_state=1)
    # Добавление нелинейного преобразования
    X_complex = X_complex @ np.array([[1, 0.5], [0.5, 1]])
    datasets['complex'] = (X_complex, y_complex)
    
    return datasets

def compare_kernels_on_datasets():
    """Сравнение различных ядер на различных наборах данных"""
    
    datasets = create_datasets()
    kernels = {
        'Linear': {'kernel': 'linear', 'C': 1.0},
        'Polynomial': {'kernel': 'poly', 'degree': 3, 'C': 1.0},
        'RBF (gamma=0.1)': {'kernel': 'rbf', 'gamma': 0.1, 'C': 1.0},
        'RBF (gamma=1.0)': {'kernel': 'rbf', 'gamma': 1.0, 'C': 1.0},
        'RBF (gamma=10.0)': {'kernel': 'rbf', 'gamma': 10.0, 'C': 1.0},
        'Sigmoid': {'kernel': 'sigmoid', 'gamma': 0.1, 'C': 1.0}
    }
    
    for dataset_name, (X, y) in datasets.items():
        print(f"\n=== Набор данных: {dataset_name.upper()} ===")
        print(f"Форма: {X.shape}, Классы: {np.unique(y)}")
        
        # Стандартизация признаков
        sc = StandardScaler()
        X_std = sc.fit_transform(X)
        
        # Создание сетки подграфиков
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Сравнение ядер SVM на наборе данных {dataset_name.capitalize()}', fontsize=16)
        
        for idx, (kernel_name, kernel_params) in enumerate(kernels.items()):
            row = idx // 3
            col = idx % 3
            ax = axes[row, col]
            
            # Train SVM
            svm = SVC(random_state=1, **kernel_params)
            svm.fit(X_std, y)
            
            # Calculate accuracy
            accuracy = svm.score(X_std, y)
            
            # Plot decision regions
            markers = ('s', 'x', 'o', '^', 'v')
            colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
            cmap = ListedColormap(colors[:len(np.unique(y))])
            
            # Decision surface
            x1_min, x1_max = X_std[:, 0].min() - 1, X_std[:, 0].max() + 1
            x2_min, x2_max = X_std[:, 1].min() - 1, X_std[:, 1].max() + 1
            xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                                   np.arange(x2_min, x2_max, 0.02))
            Z = svm.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
            Z = Z.reshape(xx1.shape)
            
            ax.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
            
            # Plot samples
            for class_idx, cl in enumerate(np.unique(y)):
                ax.scatter(X_std[y == cl, 0], X_std[y == cl, 1],
                          alpha=0.8, c=colors[class_idx], marker=markers[class_idx],
                          label=f'Class {cl}', edgecolor='black')
            
            ax.set_title(f'{kernel_name}\nAcc: {accuracy:.3f}')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            
            # Добавление опорных векторов
            if hasattr(svm, 'support_vectors_'):
                ax.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
                          s=100, linewidth=1, facecolors='none', edgecolors='yellow',
                          alpha=0.7, label='SV')
        
        plt.tight_layout()
        plt.show()
        
        # Вывод сводки
        print("\nСводка производительности ядер:")
        for kernel_name, kernel_params in kernels.items():
            svm = SVC(random_state=1, **kernel_params)
            svm.fit(X_std, y)
            accuracy = svm.score(X_std, y)
            n_support = len(svm.support_vectors_)
            print(f"{kernel_name:20}: Accuracy={accuracy:.3f}, Support Vectors={n_support}")

def demonstrate_hyperparameter_tuning():
    """Демонстрация настройки гиперпараметров для RBF ядра"""
    
    print("\n=== Настройка гиперпараметров для RBF ядра ===")
    print()
    
    # Создание сложного набора данных
    X, y = make_circles(n_samples=200, noise=0.3, factor=0.3, random_state=1)
    sc = StandardScaler()
    X_std = sc.fit_transform(X)
    
    # Сетка параметров
    C_values = [0.1, 1, 10, 100]
    gamma_values = [0.01, 0.1, 1, 10]
    
    # Создание визуализации тепловой карты
    fig, axes = plt.subplots(len(C_values), len(gamma_values), 
                            figsize=(16, 12))
    fig.suptitle('RBF SVM: Влияние параметров C и Gamma', fontsize=16)
    
    for i, C in enumerate(C_values):
        for j, gamma in enumerate(gamma_values):
            ax = axes[i, j]
            
            # Train SVM
            svm = SVC(kernel='rbf', C=C, gamma=gamma, random_state=1)
            svm.fit(X_std, y)
            
            # Calculate accuracy
            accuracy = svm.score(X_std, y)
            
            # Plot decision regions
            markers = ('s', 'x', 'o', '^', 'v')
            colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
            cmap = ListedColormap(colors[:len(np.unique(y))])
            
            # Decision surface
            x1_min, x1_max = X_std[:, 0].min() - 1, X_std[:, 0].max() + 1
            x2_min, x2_max = X_std[:, 1].min() - 1, X_std[:, 1].max() + 1
            xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                                   np.arange(x2_min, x2_max, 0.02))
            Z = svm.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
            Z = Z.reshape(xx1.shape)
            
            ax.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
            
            # Plot samples
            for class_idx, cl in enumerate(np.unique(y)):
                ax.scatter(X_std[y == cl, 0], X_std[y == cl, 1],
                          alpha=0.8, c=colors[class_idx], marker=markers[class_idx],
                          edgecolor='black')
            
            ax.set_title(f'C={C}, gamma={gamma}\nAcc: {accuracy:.3f}')
            ax.set_xlabel('Feature 1' if i == len(C_values)-1 else '')
            ax.set_ylabel('Feature 2' if j == 0 else '')
    
    plt.tight_layout()
    plt.show()
    
    # Вывод влияния параметров
    print("\nВлияние параметров:")
    print("- C (Параметр регуляризации):")
    print("  * Маленькое C: Большая маржа, больше ошибок (недообучение)")
    print("  * Большое C: Маленькая маржа, меньше ошибок (переобучение)")
    print()
    print("- gamma (Коэффициент ядра):")
    print("  * Маленькая gamma: Большой радиус сходства, более гладкая граница решений")
    print("  * Большая gamma: Маленький радиус сходства, более сложная граница")

def kernel_trick_practical_tips():
    """Предоставление практических советов по использованию ядерного трюка"""
    
    print("\n=== Практические советы по использованию ядерного трюка ===")
    print()
    
    tips = [
        "1. Начните с RBF ядра - оно универсально и хорошо работает в большинстве случаев",
        "2. Используйте StandardScaler перед применением ядер (особенно RBF)",
        "3. Настройте gamma и C для RBF ядра с помощью кросс-валидации",
        "4. Линейное ядро достаточно для данных высокой размерности (текст, геномика)",
        "5. Полиномиальное ядро полезно, когда важны взаимодействия признаков",
        "6. Следите за переобучением при сложных ядрах и большом C",
        "7. Учитывайте вычислительные затраты: RBF > Полиномиальное > Линейное",
        "8. Используйте количество опорных векторов для оценки сложности модели",
        "9. Для больших наборов данных используйте LinearSVC или методы аппроксимации",
        "10. Всегда визуализируйте границы решений, когда возможно"
    ]
    
    for tip in tips:
        print(tip)
    
    print()
    print("Руководство по выбору параметров:")
    print("- Если модель недообучается: Увеличить C, увеличить gamma (для RBF)")
    print("- Если модель переобучается: Уменьшить C, уменьшить gamma (для RBF)")
    print("- Если обучение медленное: Использовать линейное ядро или уменьшить C")
    print("- Если слишком много опорных векторов: Увеличить C, настроить gamma")

if __name__ == "__main__":
    compare_kernels_on_datasets()
    demonstrate_hyperparameter_tuning()
    kernel_trick_practical_tips()
