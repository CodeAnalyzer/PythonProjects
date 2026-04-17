import numpy as np

class AdalineGD:
    """Классификатор на адаптивных линейных нейронах.
    
    Параметры
    ----------
    eta : float
        Скорость обучения (между 0.0 и 1.0)
    n_iter : int
        Количество проходов по обучающему набору.
    random_state : int
        Начальное значение для случайной инициализации весов.
    
    Атрибуты
    ----------
    w_ : 1d-array
        Веса после обучения.
    b_ : Scalar
        Смещение после обучения.
    losses_ : list
        Значения среднеквадратичной функции потерь после каждой эпохи.
    """
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        """Подгонка к обучающим данным.
        
        Параметры
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Обучающие векторы, где n_examples - количество образцов и
            n_features - количество признаков.
        y : array-like, shape = [n_examples]
            Целевые переменные.
        
        Возвращаемые значения
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.)  # Исправлено для NumPy 2.0
        self.losses_ = []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            
            # Градиентный спуск
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            
            # Среднеквадратичная ошибка
            loss = (errors**2).mean()
            self.losses_.append(loss)
            
            # Отладочная информация каждые 100 эпох
            if (i + 1) % 100 == 0:
                print(f"Эпоха {i+1}: ошибка={loss:.6f}, веса={self.w_}, смещение={self.b_:.6f}")
            
        return self
    
    def net_input(self, X):
        """Вычисление фактического ввода"""
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, X):
        """Вычисление линейной активации"""
        return X
    
    def predict(self, X):
        """Возвращаем метку класса"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, 0)
