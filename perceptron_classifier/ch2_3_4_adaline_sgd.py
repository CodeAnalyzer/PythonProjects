import numpy as np

class AdalineSGD:
    """Классификатор на адаптивных линейных нейронах со стохастическим градиентным спуском.
    
    Параметры
    ----------
    eta : float
        Скорость обучения (между 0.0 и 1.0).
    n_iter : int
        Количество проходов по обучающему набору.
    shuffle : bool (default: True)
        Перемешивание обучающих данных каждую эпоху, если задано True,
        для предотвращения возникновения циклов.
    random_state : int
        Затравка генератора случайных чисел для инициализации весов
        случайными значениями.
    
    Атрибуты
    ----------
    w_ : 1d-array
        Веса после обучения.
    b_ : Scalar
        Смещение после обучения.
    losses_ : list
        Значения среднеквадратичной функции потерь после каждой эпохи.
    """
    
    def __init__(self, eta=0.01, n_iter=10, shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.w_initialized = False
    
    def fit(self, X, y):
        """Подгонка к обучающим данным.
        
        Параметры
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Обучающие векторы, где n_examples - количество образцов,
            а n_features - количество признаков.
        y : array-like, shape = [n_examples]
            Целевые переменные.
        
        Возвращаемые значения
        -------
        self : object
        """
        self._initialize_weights(X.shape[1])
        self.losses_ = []
        
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            
            losses = []
            for xi, target in zip(X, y):
                losses.append(self._update_weights(xi, target))
            
            avg_loss = np.mean(losses)
            self.losses_.append(avg_loss)
            
            # Отладочная информация
            if i % 5 == 0 or i == self.n_iter - 1:
                print(f"Эпоха {i+1}: средняя потеря={avg_loss:.6f}, "
                      f"веса=[{self.w_[0]:.6f} {self.w_[1]:.6f}], "
                      f"смещение={self.b_:.6f}")
        
        return self
    
    def partial_fit(self, X, y):
        """Подгонка к обучающим данным без повторной инициализации весов"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        
        return self
    
    def _shuffle(self, X, y):
        """Перемешивание обучающих данных"""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """Инициализация весов небольшими случайными значениями"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float64(0.0)
        self.w_initialized = True
    
    def _update_weights(self, xi, target):
        """Применение правила Adaline для обновления весов"""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        
        # Обновление весов и смещения
        self.w_ += self.eta * 2.0 * xi * (error)
        self.b_ += self.eta * 2.0 * error
        
        loss = error**2
        return loss
    
    def net_input(self, X):
        """Вычисление фактического ввода"""
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, X):
        """Вычисление линейной активации"""
        return X
    
    def predict(self, X):
        """Возвращаем метку класса"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
