"""
Глава 14: Глубокое обучение с использованием сверточных нейронных сетей
Раздел 14.1.2: Выполнение дискретных сверток
Вычисление дискретной двумерной свертки вручную и с помощью scipy
"""

import numpy as np
import scipy.signal


def conv2d(X, W, p=(0, 0), s=(1, 1)):
    """
    Ручная реализация двумерной свертки
    
    Parameters:
    -----------
    X : array-like
        Входная матрица (изображение/признаки)
    W : array-like
        Ядро свертки (фильтр/веса)
    p : tuple(int, int)
        Паддинг (padding) по каждой оси
    s : tuple(int, int)  
        Шаг (stride) по каждой оси
    
    Returns:
    --------
    np.array : Результат свертки
    """
    # Вращаем ядро на 180 градусов (для корректной свертки)
    W_rot = np.array(W)[::-1, ::-1]
    X_orig = np.array(X)
    
    # Размеры с паддингом
    n1 = X_orig.shape[0] + 2 * p[0]
    n2 = X_orig.shape[1] + 2 * p[1]
    
    # Создаем матрицу с паддингом (заполнена нулями)
    X_padded = np.zeros(shape=(n1, n2))
    X_padded[p[0]:p[0] + X_orig.shape[0], p[1]:p[1] + X_orig.shape[1]] = X_orig
    
    # Вычисляем размер выходной матрицы
    res = []
    for i in range(0, int((X_padded.shape[0] - W_rot.shape[0]) / s[0]) + 1, s[0]):
        res.append([])
        for j in range(0, int((X_padded.shape[1] - W_rot.shape[1]) / s[1]) + 1, s[1]):
            # Извлекаем подматрицу
            X_sub = X_padded[i:i + W_rot.shape[0], j:j + W_rot.shape[1]]
            # Поэлементное умножение и суммирование
            res[-1].append(np.sum(X_sub * W_rot))
    
    return np.array(res)


def main():
    # Тестовые данные из книги
    X = np.array([[1, 3, 2, 4], 
                  [5, 6, 1, 3], 
                  [1, 2, 0, 2], 
                  [3, 4, 3, 2]])
    
    W = np.array([[1, 0, 3], 
                  [1, 2, 1], 
                  [0, 1, 1]])
    
    print("=" * 60)
    print("Двумерная дискретная свертка")
    print("=" * 60)
    
    print("\nВходная матрица X:")
    print(X)
    
    print("\nЯдро свертки W:")
    print(W)
    
    # Ручная реализация
    result_manual = conv2d(X, W, p=(1, 1), s=(1, 1))
    print("\n" + "-" * 40)
    print("Реализация Conv2d (ручная):")
    print("-" * 40)
    print(result_manual)
    
    # SciPy реализация
    result_scipy = scipy.signal.convolve2d(X, W, mode='same')
    print("\n" + "-" * 40)
    print("Результат SciPy (mode='same'):")
    print("-" * 40)
    print(result_scipy)
    
    # Проверка совпадения
    print("\n" + "=" * 60)
    print("Проверка совпадения результатов:")
    print(f"Результаты идентичны: {np.allclose(result_manual, result_scipy)}")
    print("=" * 60)
    
    # Дополнительно: различные режимы свертки в SciPy
    print("\n" + "-" * 40)
    print("Режимы свертки SciPy:")
    print("-" * 40)
    
    # 'full' - полная свертка
    result_full = scipy.signal.convolve2d(X, W, mode='full')
    print(f"\nmode='full' (размер {result_full.shape}):")
    print(result_full)
    
    # 'valid' - только где фильтр полностью накладывается
    result_valid = scipy.signal.convolve2d(X, W, mode='valid')
    print(f"\nmode='valid' (размер {result_valid.shape}):")
    print(result_valid)
    
    # 'same' - сохраняет размер входа
    result_same = scipy.signal.convolve2d(X, W, mode='same')
    print(f"\nmode='same' (размер {result_same.shape}):")
    print(result_same)
    
    # Демонстрация различных stride
    print("\n" + "=" * 60)
    print("Влияние stride (шага) на результат:")
    print("=" * 60)
    
    for stride in [(1, 1), (2, 2)]:
        result = conv2d(X, W, p=(0, 0), s=stride)
        print(f"\nstride={stride}, размер выхода: {result.shape}")
        print(result)
    
    # Демонстрация различных padding
    print("\n" + "=" * 60)
    print("Влияние padding на результат:")
    print("=" * 60)
    
    for pad in [(0, 0), (1, 1), (2, 2)]:
        result = conv2d(X, W, p=pad, s=(1, 1))
        print(f"\npadding={pad}, размер выхода: {result.shape}")
        print(result)


if __name__ == "__main__":
    main()
