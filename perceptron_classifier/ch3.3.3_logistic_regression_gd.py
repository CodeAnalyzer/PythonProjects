import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

class LogisticRegressionGD:
    """Классификатор на основе логистической регрессии с использованием градиентного спуска.
    
    Параметры
    ----------
    eta : float
        Скорость обучения (между 0.0 и 1.0).
    n_iter : int
        Количество проходов по обучающему набору данных.
    random_state : int
        Затравка генератора случайных чисел для инициализации весов.
        
    Атрибуты
    ----------
    w_ : 1d-array
        Веса после обучения.
    b_ : Scalar
        Смещение после обучения.
    losses_ : list
        Значения функции потерь в каждой эпохе.
    """
    
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
    
    def fit(self, X, y):
        """Подгонка под обучающие данные.
        
        Параметры
        ----------
        X : {array-like}, shape = [n_examples, n_features]
            Обучающие векторы, где n_examples - количество примеров,
            а n_features - количество признаков.
        y : array-like, shape = [n_examples]
            Целевые значения.
            
        Возвращает
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=X.shape[1])
        self.b_ = np.float64(0.)
        self.losses_ = []
        
        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_ += self.eta * 2.0 * X.T.dot(errors) / X.shape[0]
            self.b_ += self.eta * 2.0 * errors.mean()
            loss = (-y.dot(np.log(output)) - 
                   ((1 - y).dot(np.log(1 - output)))) / X.shape[0]
            self.losses_.append(loss)
            
        return self
    
    def net_input(self, X):
        """Вычислить чистый вход"""
        return np.dot(X, self.w_) + self.b_
    
    def activation(self, z):
        """Вычислить логистическую сигмоидную активацию"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))
    
    def predict(self, X):
        """Вернуть метку класса после порогового шага"""
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)


def plot_decision_regions(X, y, classifier, resolution=0.02):
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
    
    # построение примеров классов
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=color_list[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')


# Example usage with iris dataset (for demonstration)
if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    # Загрузить набор данных iris
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]  # длина и ширина лепестка
    y = iris.target
    
    # Использовать только setosa и versicolor (классы 0 и 1)
    X = X[y != 2]
    y = y[y != 2]
    
    # Разделить данные
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1, stratify=y)
    
    # Стандартизировать признаки
    sc = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std = sc.transform(X_test)
    
    # Обучить логистическую регрессию
    lrgd = LogisticRegressionGD(eta=0.3, n_iter=1000, random_state=1)
    lrgd.fit(X_train_std, y_train)
    
    # Построить области принятия решений
    plot_decision_regions(X=X_train_std, y=y_train, classifier=lrgd)
    plt.xlabel('Длина лепестка [стандартизирована]')
    plt.ylabel('Ширина лепестка [стандартизирована]')
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()
    
    # Сделать прогнозы
    y_pred = lrgd.predict(X_test_std)
    accuracy = np.mean(y_pred == y_test)
    print(f'Точность: {accuracy:.3f}')
