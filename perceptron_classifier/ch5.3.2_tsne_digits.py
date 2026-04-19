"""
Раздел 5.3.2. Визуализация данных с помощью алгоритма t-SNE
Учебный пример из книги "Python Machine Learning"
"""

import matplotlib.patheffects as PathEffects
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE


print("=" * 72)
print("Раздел 5.3.2. Визуализация данных с помощью t-SNE")
print("=" * 72)


# Загрузка набора данных Digits
print("\n=== 1. Загрузка набора данных Digits ===")
digits = load_digits()

print(f"Количество изображений: {len(digits.images)}")
print(f"Размер каждого изображения: {digits.images[0].shape}")
print(f"Размер набора данных (data): {digits.data.shape}")
print(f"Классы (цифры): {np.unique(digits.target)}")

# Отображение первых 4 изображений
print("\n=== 2. Отображение первых 4 изображений ===")
fig, ax = plt.subplots(1, 4, figsize=(12, 3))
for i in range(4):
    ax[i].imshow(digits.images[i], cmap='Greys')
    ax[i].set_title(f'Цифра: {digits.target[i]}')
    ax[i].axis('off')
plt.tight_layout()
plt.savefig('d:\\GITHUB\\PythonProjects\\perceptron_classifier\\tsne_digits_first4.png', dpi=150)
print("График первых 4 изображений сохранен как: tsne_digits_first4.png")
plt.show()

# Подготовка данных
print("\n=== 3. Подготовка данных ===")
y_digits = digits.target
X_digits = digits.data

print(f"Форма X_digits: {X_digits.shape}")
print(f"Форма y_digits: {y_digits.shape}")

# Применение t-SNE
print("\n=== 4. Применение t-SNE для уменьшения размерности ===")
tsne = TSNE(n_components=2, init='pca', random_state=123)
X_digits_tsne = tsne.fit_transform(X_digits)

print(f"Форма X_digits_tsne после t-SNE: {X_digits_tsne.shape}")
print("Первые 5 преобразованных образцов:")
print(np.round(X_digits_tsne[:5], 4))


# Функция для визуализации проекции с метками классов
def plot_projection(x, colors):
    """
    Визуализация 2D-проекции с метками классов в центре каждой группы
    
    Параметры:
    -----------
    x : array-like, shape (n_samples, 2)
        Двумерные координаты точек после t-SNE
    colors : array-like, shape (n_samples,)
        Метки классов для раскраски точек
    """
    f = plt.figure(figsize=(8, 8))
    ax = plt.subplot(aspect='equal')
    
    # Цвета для 10 классов (цифры 0-9)
    colors_map = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for i in range(10):
        # Отображение точек для каждого класса
        plt.scatter(
            x[colors == i, 0],
            x[colors == i, 1],
            c=[colors_map[i]],
            label=f'Цифра {i}',
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5
        )
        
        # Добавление метки класса в центре группы
        xtext, ytext = np.median(x[colors == i, :], axis=0)
        txt = ax.text(xtext, ytext, str(i), fontsize=24)
        txt.set_path_effects([
            PathEffects.Stroke(linewidth=5, foreground="w"),
            PathEffects.Normal()
        ])
    
    plt.xlabel('t-SNE компонента 1')
    plt.ylabel('t-SNE компонента 2')
    plt.title('Визуализация рукописных цифр с помощью t-SNE')
    plt.tight_layout()
    return f


# Визуализация результатов t-SNE
print("\n=== 5. Визуализация 2D-встраиваний t-SNE ===")
fig = plot_projection(X_digits_tsne, y_digits)
plt.savefig('d:\\GITHUB\\PythonProjects\\perceptron_classifier\\tsne_digits_projection.png', dpi=150)
print("График t-SNE проекции сохранен как: tsne_digits_projection.png")
plt.show()

# Дополнительный анализ: сравнение с PCA
print("\n=== 6. Сравнение с PCA (для справки) ===")
from sklearn.decomposition import PCA

pca = PCA(n_components=2, random_state=123)
X_digits_pca = pca.fit_transform(X_digits)

print(f"Форма X_digits_pca: {X_digits_pca.shape}")
print("Объясненная дисперсия PCA:")
print(f"PC1: {pca.explained_variance_ratio_[0]:.4f}")
print(f"PC2: {pca.explained_variance_ratio_[1]:.4f}")
print(f"Суммарная: {pca.explained_variance_ratio_.sum():.4f}")

# Визуализация PCA для сравнения
fig_pca = plot_projection(X_digits_pca, y_digits)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Визуализация рукописных цифр с помощью PCA')
plt.savefig('d:\\GITHUB\\PythonProjects\\perceptron_classifier\\pca_digits_projection.png', dpi=150)
print("График PCA проекции сохранен как: pca_digits_projection.png")
plt.show()

# Сравнительный анализ
print("\n=== 7. Сравнительный анализ ===")
print("t-SNE лучше разделяет классы, чем PCA, потому что:")
print("- t-SNE сохраняет локальную структуру данных")
print("- t-SNE способен захватывать нелинейные зависимости")
print("- PCA является линейным методом и сохраняет только глобальную дисперсию")
print("\nВажно помнить:")
print("- t-SNE - это метод визуализации, не для обучения моделей")
print("- t-SNE требует весь набор данных для проекции")
print("- t-SNE нельзя применить к новым точкам данных без повторного обучения")
print("- Гиперпараметры t-SNE (perplexity, learning rate) могут влиять на результаты")

print("\n" + "=" * 72)
print("Раздел 5.3.2 завершен")
print("=" * 72)
