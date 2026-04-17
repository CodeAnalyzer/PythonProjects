import numpy as np

class Perceptron:
    """Персептронный классификатор.
    
    Параметры
    ----------
    eta : float
        Скорость обучения (между 0.0 и 1.0)
    n_iter : int
        Кол-во проходов по обучающему набору.
    random_state : int
        Опорное значение генератора случайных чисел для инициализации весов.
    
    Атрибуты
    ----------
    w_ : 1d-array
        Веса после подгонки.
    b_ : Scalar
        Смещение после подгонки.
    errors_ : list
        Количество неправильных классификаций (обновлений) в каждой эпохе.
    """
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        """Соответствие тренировочным данным.
        
        Параметры
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Обучающий вектор, где n_examples - это количество образцов,
            а n_features - количество признаков.
        y : array-like, shape = [n_examples]
            Целевые значения.
        
        Возвращаемые значения
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.)
        self.errors_ = []
        
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_ += update * xi
                self.b_ += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self, X):
        """Вычисление фактического входа"""
        return np.dot(X, self.w_) + self.b_
    
    def predict(self, X):
        """Возвращает метки класса после шага"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
