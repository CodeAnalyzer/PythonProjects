import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    """Построить области принятия решений"""
    # настройка генератора маркеров и цветовой карты
    markers = ('s', 'x', 'o', '^', 'v')
    color_list = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = colors.ListedColormap(color_list[:len(np.unique(y))])
    
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
    
    # построение всех примеров
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=color_list[idx],
                    marker=markers[idx], 
                    label=f'Класс {cl}',
                    edgecolor='black')
    
    # выделение тестовых примеров
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    facecolors='none', edgecolor='black', alpha=1.0,
                    linewidth=1, marker='o',
                    s=100, label='тестовый набор')

# Загрузить набор данных iris
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]  # длина и ширина лепестка
y = iris.target

# Разделить данные
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

# Стандартизировать признаки
sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# Объединить обучающие и тестовые данные для визуализации
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# Создать и обучить логистическую регрессию из scikit-learn
lr = LogisticRegression(C=100.0, solver='lbfgs', random_state=1)
lr.fit(X_train_std, y_train)

# Построить области принятия решений
plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105, 150))
plt.xlabel('Длина лепестка [стандартизирована]')
plt.ylabel('Ширина лепестка [стандартизирована]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Сделать прогнозы и вычислить точность
y_pred = lr.predict(X_test_std)
accuracy = np.mean(y_pred == y_test)
print(f'Точность: {accuracy:.3f}')

# Показать вероятности для первого тестового образца
print(f'\n=== Вероятности для первого тестового образца ===')
probabilities = lr.predict_proba(X_test_std[0].reshape(1, -1))[0]
print(f'Вероятности: [Класс 0: {probabilities[0]:.6f}, Класс 1: {probabilities[1]:.6f}, Класс 2: {probabilities[2]:.6f}]')
print(f'Предсказанный класс: {lr.predict(X_test_std[0].reshape(1, -1))[0]}')
print(f'Истинный класс: {y_test[0]}')

# Показать вероятности для первых 5 тестовых образцов
print(f'\n=== Вероятности для первых 5 тестовых образцов ===')
for i in range(5):
    probs = lr.predict_proba(X_test_std[i].reshape(1, -1))[0]
    predicted = lr.predict(X_test_std[i].reshape(1, -1))[0]
    true_class = y_test[i]
    print(f'Образец {i+1}: [0: {probs[0]:.4f}, 1: {probs[1]:.4f}, 2: {probs[2]:.4f}] -> Предсказанный: {predicted}, Истинный: {true_class}')

# Сравнить с мультиномиальным подходом
print("\n" + "="*50)
print("Сравнение с мультиномиальным подходом:")
lr_multinomial = LogisticRegression(C=100.0, solver='lbfgs', random_state=1)
lr_multinomial.fit(X_train_std, y_train)
y_pred_multi = lr_multinomial.predict(X_test_std)
accuracy_multi = np.mean(y_pred_multi == y_test)
print(f'Точность мультиномиального подхода: {accuracy_multi:.3f}')
print(f'Точность OvR подхода: {accuracy:.3f}')
