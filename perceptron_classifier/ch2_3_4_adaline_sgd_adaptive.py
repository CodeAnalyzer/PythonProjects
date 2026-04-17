import numpy as np

class AdalineSGDAdaptive:
    """Классификатор на адаптивных линейных нейронах со стохастическим градиентным спуском
    и адаптивной скоростью обучения.
    
    Параметры
    ----------
    eta0 : float
        Начальная скорость обучения (между 0.0 и 1.0).
    n_iter : int
        Количество проходов по обучающему набору.
    shuffle : bool (default: True)
        Перемешивание обучающих данных каждую эпоху.
    random_state : int
        Затравка генератора случайных чисел.
    adaptive : bool (default: True)
        Использовать адаптивную скорость обучения.
    c1 : float (default: 1000.0)
        Константа для адаптивной скорости обучения.
    c2 : float (default: 1.0)
        Константа для адаптивной скорости обучения.
    
    Атрибуты
    ----------
    w_ : 1d-array
        Веса после обучения.
    b_ : Scalar
        Смещение после обучения.
    losses_ : list
        Значения среднеквадратичной функции потерь после каждой эпохи.
    eta_history_ : list
        История изменения скорости обучения.
    """
    
    def __init__(self, eta0=0.01, n_iter=10, shuffle=True, random_state=None,
                 adaptive=True, c1=1000.0, c2=1.0):
        self.eta0 = eta0  # Начальная скорость обучения
        self.n_iter = n_iter
        self.shuffle = shuffle
        self.random_state = random_state
        self.w_initialized = False
        self.adaptive = adaptive
        self.c1 = c1
        self.c2 = c2
        self.eta_history_ = []
    
    def _get_eta(self, iteration):
        """Вычисление адаптивной скорости обучения"""
        if self.adaptive:
            # Адаптивная формула: eta = c1 / (iteration + c2)
            eta = self.c1 / (iteration + self.c2)
            # Ограничиваем максимальное значение начальной скоростью
            return min(eta, self.eta0)
        else:
            return self.eta0
    
    def fit(self, X, y):
        """Подгонка к обучающим данным с адаптивной скоростью обучения."""
        self._initialize_weights(X.shape[1])
        self.losses_ = []
        self.eta_history_ = []
        
        total_iterations = 0  # Общее количество итераций
        
        for epoch in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            
            losses = []
            etas_in_epoch = []
            
            for xi, target in zip(X, y):
                # Получаем адаптивную скорость обучения
                eta = self._get_eta(total_iterations)
                etas_in_epoch.append(eta)
                
                # Обновляем веса с текущей скоростью обучения
                loss = self._update_weights(xi, target, eta)
                losses.append(loss)
                
                total_iterations += 1
            
            # Сохраняем статистику
            avg_loss = np.mean(losses)
            avg_eta = np.mean(etas_in_epoch)
            
            self.losses_.append(avg_loss)
            self.eta_history_.append(avg_eta)
            
            # Отладочная информация
            if epoch % 5 == 0 or epoch == self.n_iter - 1:
                print(f"Эпоха {epoch+1}: потеря={avg_loss:.6f}, "
                      f"eta={avg_eta:.6f}, "
                      f"веса=[{self.w_[0]:.6f} {self.w_[1]:.6f}], "
                      f"смещение={self.b_:.6f}")
        
        return self
    
    def partial_fit(self, X, y):
        """Подгонка к обучающим данным без повторной инициализации весов."""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                eta = self._get_eta(len(self.eta_history_) * len(X))
                self._update_weights(xi, target, eta)
        else:
            eta = self._get_eta(len(self.eta_history_) * len(X))
            self._update_weights(X, y, eta)
        
        return self
    
    def _shuffle(self, X, y):
        """Перемешивание обучающих данных."""
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initialize_weights(self, m):
        """Инициализация весов небольшими случайными значениями."""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=m)
        self.b_ = np.float64(0.0)
        self.w_initialized = True
    
    def _update_weights(self, xi, target, eta):
        """Применение правила Adaline для обновления весов с заданной eta."""
        output = self.activation(self.net_input(xi))
        error = (target - output)
        
        # Обновление весов с адаптивной скоростью обучения
        self.w_ += eta * 2.0 * xi * (error)
        self.b_ += eta * 2.0 * error
        
        loss = error**2
        return loss
    
    def net_input(self, X):
        """Вычисление фактического ввода."""
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, X):
        """Вычисление линейной активации."""
        return X
    
    def predict(self, X):
        """Возвращаем метку класса."""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)
