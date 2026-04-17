import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

print("⚖️ ПУТЬ РЕГУЛЯРИЗАЦИИ L2 ДЛЯ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ")
print("=" * 60)

# 1. Загрузка и подготовка данных
print("\n📊 1. ЗАГРУЗКА ДАННЫХ IRIS")
print("-" * 40)
iris = datasets.load_iris()
X = iris.data[:, [2, 3]]  # Длина и ширина лепестков
y = iris.target

print(f'Классы: {iris.target_names}')
print(f'Признаки: {iris.feature_names[2:4]}')

# Используем только setosa и versicolor для бинарной классификации
X = X[y != 2]  # Убираем virginica
y = y[y != 2]  # Убираем virginica

print(f'После фильтрации (только setosa и versicolor):')
print(f'Форма X: {X.shape}')
print(f'Классы: {np.unique(y, return_counts=True)}')

# 2. Разделение данных
print("\n🔀 2. РАЗДЕЛЕНИЕ ДАННЫХ")
print("-" * 40)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

print(f'Обучающий набор: {X_train.shape[0]} образцов')
print(f'Тестовый набор: {X_test.shape[0]} образцов')

# 3. Стандартизация
print("\n⚖️ 3. СТАНДАРТИЗАЦИЯ ПРИЗНАКОВ")
print("-" * 40)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

print(f'Средние значения (стандартизированные): [{X_train_std.mean(axis=0)[0]:.6f}, {X_train_std.mean(axis=0)[1]:.6f}]')
print(f'Стандартные отклонения (стандартизированные): [{X_train_std.std(axis=0)[0]:.6f}, {X_train_std.std(axis=0)[1]:.6f}]')

# 4. Путь регуляризации L2
print("\n🧠 4. ПОСТРОЕНИЕ ПУТИ РЕГУЛЯРИЗАЦИИ L2")
print("-" * 40)
print("Обучение 10 моделей логистической регрессии с разными значениями C")
print("(C = 10^c, где c от -5 до 4)")

weights, params = [], []

for c in np.arange(-5, 5):
    # Обучение модели с текущим значением C
    # C = 10^c - обратный параметр регуляризации
    lr = LogisticRegression(C=10.**c, solver='lbfgs', random_state=1)
    lr.fit(X_train_std, y_train)
    
    # Сохраняем веса для класса 1 (versicolor)
    # В бинарной классификации lr.coef_[0] содержит веса для положительного класса
    weights.append(lr.coef_[0])
    params.append(10.**c)

weights = np.array(weights)
print(f"\nПолучено {len(weights)} наборов весов")
print(f"Форма массива весов: {weights.shape}")

# 5. Визуализация пути регуляризации
print("\n📈 5. ВИЗУАЛИЗАЦИЯ ПУТИ РЕГУЛЯРИЗАЦИИ")
print("-" * 40)

plt.figure(figsize=(10, 6))
plt.plot(params, weights[:, 0], 
         label='Длина лепестка', 
         linewidth=2, markersize=8, marker='o')
plt.plot(params, weights[:, 1], 
         linestyle='--',
         label='Ширина лепестка', 
         linewidth=2, markersize=8, marker='s')

plt.ylabel('Весовой коэффициент', fontsize=12)
plt.xlabel('C (обратный параметр регуляризации)', fontsize=12)
plt.title('Путь регуляризации L2 для логистической регрессии', fontsize=14, fontweight='bold')
plt.legend(loc='upper left', fontsize=10)
plt.xscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 6. Анализ результатов
print("\n🔍 6. АНАЛИЗ РЕЗУЛЬТАТОВ")
print("-" * 40)

print(f"{'C':<12} {'log10(C)':<12} {'Вес (длина)':<15} {'Вес (ширина)':<15}")
print("-" * 60)

for i, (param, weight) in enumerate(zip(params, weights)):
    print(f"{param:<12.3e} {np.log10(param):<12.1f} {weight[0]:<15.3f} {weight[1]:<15.3f}")

# 7. Теоретическое объяснение
print("\n📚 7. ТЕОРЕТИЧЕСКОЕ ОБЪЯСНЕНИЕ")
print("-" * 40)

print("РЕГУЛЯРИЗАЦИЯ L2 В ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ")
print("\nФункция потерь с регуляризацией:")
print("L(w,b) = -Σ[y(i)log(σ(z(i))) + (1-y(i))log(1-σ(z(i)))] + (λ/2)||w||²")
print("\nЧастная производная с регуляризацией:")
print("∂L/∂w = -(1/n)Σ(y(i) - σ(w^T x(i)))x(i) + λw")

print("\nПАРАМЕТР C:")
print("• C = 1/λ (обратный параметр регуляризации)")
print("• Большое C → малая λ → слабая регуляризация")
print("• Малое C → большая λ → сильная регуляризация")

print("\nИНТЕРПРЕТАЦИЯ ГРАФИКА:")
print("• Малые значения C (слева): сильная регуляризация → веса близки к 0")
print("• Большие значения C (справа): слабая регуляризация → веса растут")
print("• Оптимальное C: баланс между смещением и дисперсией")

print("\nПОЧЕМУ ВЕСА УМЕНЬШАЮТСЯ:")
print("• Регуляризация штрафует большие веса")
print("• L2 норма: ||w||² = w₁² + w₂² + ... + wₙ²")
print("• Модель предпочитает маленькие веса для минимизации функции потерь")

# 8. Практические рекомендации
print("\n💡 8. ПРАКТИЧЕСКИЕ РЕКОМЕНДАЦИИ")
print("-" * 40)

print("ВЫБОР ПАРАМЕТРА C:")
print("• C < 0.01: слишком сильная регуляризация (недообучение)")
print("• C = 0.01-0.1: сильная регуляризация (высокое смещение)")
print("• C = 1.0: умеренная регуляризация (хороший баланс)")
print("• C = 10-100: слабая регуляризация (риск переобучения)")
print("• C > 100: очень слабая регуляризация (переобучение)")

print("\nСТРАТЕГИИ ВЫБОРА:")
print("1. Grid Search: перебор значений C на сетке")
print("2. Cross Validation: оценка качества на разных фолдах")
print("3. Анализ графика: поиск плато весов")

print("\n🎯 ВЫВОДЫ:")
print("=" * 60)
print("• Путь регуляризации показывает, как веса меняются с C")
print("• Сильная регуляризация (малый C) → веса → 0")
print("• Слабая регуляризация (большой C) → веса растут")
print("• Оптимальное C находится в средней области графика")
print("• Регуляризация предотвращает переобучение")
