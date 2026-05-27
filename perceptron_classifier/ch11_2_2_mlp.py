"""
Глава 11.2.2. Реализация многослойного персептрона
Глава 11.2.3. Обучающий цикл нейронной сети
"""

import numpy as np
import matplotlib.pyplot as plt

# Вспомогательные функции для MLP
def sigmoid(z):
    """Логистическая сигмоидная функция активации"""
    return 1. / (1. + np.exp(-z))

def int_to_onehot(y, num_labels):
    """Преобразование массива меток целочисленного класса в метки с унитарным кодированием"""
    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):
        ary[i, val] = 1
    return ary

# Класс многослойного персептрона
class NeuralNetMLP:
    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        super().__init__()
        self.num_classes = num_classes
        
        # Скрытый слой
        rng = np.random.RandomState(random_seed)
        self.weight_h = rng.normal(
            loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)
        
        # Выходной слой
        self.weight_out = rng.normal(
            loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)
    
    def forward(self, x):
        """Прямой проход через сеть"""
        # Скрытый слой
        # размерность входа: [n_examples, n_features]
        # dot [n_hidden, n_features].T
        # размерность выхода: [n_examples, n_hidden]
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)
        
        # Выходной слой
        # размерность входа: [n_examples, n_hidden]
        # dot [n_classes, n_hidden].T
        # размерность выхода: [n_examples, n_classes]
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)
        
        return a_h, a_out
    
    def backward(self, x, a_h, a_out, y):
        """Обратное распространение ошибки - вычисление градиентов"""
        #########################
        ### Веса ВЫХОДНОГО слоя
        #########################
        # унитарное кодирование
        y_onehot = int_to_onehot(y, self.num_classes)
        
        # Часть 1: dLoss/dOutWeights
        # = dLoss/dOutAct * dOutAct/dOutNet * dOutNet/dOutWeight
        # где DeltaOut = dLoss/dOutAct * dOutAct/dOutNet
        # для удобства повторного использования
        
        # размер входа/выхода: [n_examples, n_classes]
        d_loss__d_a_out = 2.*(a_out - y_onehot) / y.shape[0]
        
        # размер входа/выхода: [n_examples, n_classes]
        d_a_out__d_z_out = a_out * (1. - a_out)  # сигмоидная производная
        
        # размер выхода: [n_examples, n_classes]
        delta_out = d_loss__d_a_out * d_a_out__d_z_out
        
        # градиент для выходных весов
        # [n_examples, n_hidden]
        d_z_out__dw_out = a_h
        
        # размер входа: [n_classes, n_examples] dot [n_examples, n_hidden]
        # размер выхода: [n_classes, n_hidden]
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)
        
        #################################
        ### Часть 2: dLoss/dHiddenWeights
        # = DeltaOut * dOutNet/dHiddenAct * dHiddenAct/dHiddenNet * dHiddenNet/dWeight
        #################################
        
        # [n_classes, n_hidden]
        d_z_out__a_h = self.weight_out
        
        # размер выхода: [n_examples, n_hidden]
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)
        
        # [n_examples, n_hidden]
        d_a_h__d_z_h = a_h * (1. - a_h)  # сигмоидная производная
        
        # [n_examples, n_features]
        d_z_h__dw_h = x
        
        # размер выхода: [n_hidden, n_features]
        d_loss__dw_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__dw_h)
        d_loss__db_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)
        
        return (d_loss__dw_out, d_loss__db_out,
                d_loss__dw_h, d_loss__db_h)

# Генератор мини-пакетов
def minibatch_generator(X, y, minibatch_size):
    """Генератор мини-пакетов для стохастического градиентного спуска"""
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    for start_idx in range(0, indices.shape[0] - minibatch_size + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        yield X[batch_idx], y[batch_idx]

# Функции потерь и метрик
def mse_loss(targets, probas, num_labels=10):
    """Среднеквадратичная ошибка"""
    onehot_targets = int_to_onehot(targets, num_labels=num_labels)
    return np.mean((onehot_targets - probas)**2)

def accuracy(targets, predicted_labels):
    """Точность классификации"""
    return np.mean(predicted_labels == targets)

def compute_mse_and_acc(nnet, X, y, num_labels=10, minibatch_size=100):
    """Вычисление MSE и точности с использованием мини-пакетов"""
    mse, correct_pred, num_examples = 0., 0, 0
    minibatch_gen = minibatch_generator(X, y, minibatch_size)
    for i, (features, targets) in enumerate(minibatch_gen):
        _, probas = nnet.forward(features)
        predicted_labels = np.argmax(probas, axis=1)
        
        onehot_targets = int_to_onehot(targets, num_labels=num_labels)
        loss = np.mean((onehot_targets - probas)**2)
        correct_pred += (predicted_labels == targets).sum()
        num_examples += targets.shape[0]
        mse += loss
    
    mse = mse / i
    acc = correct_pred / num_examples
    return mse, acc

# Функция обучения
def train(model, X_train, y_train, X_valid, y_valid, num_epochs, 
          minibatch_size=100, learning_rate=0.1):
    """Обучающий цикл нейронной сети"""
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []
    
    for e in range(num_epochs):
        # Итерация по мини-пакетам
        minibatch_gen = minibatch_generator(
            X_train, y_train, minibatch_size)
        
        for X_train_mini, y_train_mini in minibatch_gen:
            #### Вычисление выходов ####
            a_h, a_out = model.forward(X_train_mini)
            
            #### Вычисление градиентов ####
            d_loss__dw_out, d_loss__db_out, \
            d_loss__dw_h, d_loss__db_h = \
                model.backward(X_train_mini, a_h, a_out, y_train_mini)
            
            #### Обновление весов ####
            model.weight_h -= learning_rate * d_loss__dw_h
            model.bias_h -= learning_rate * d_loss__db_h
            model.weight_out -= learning_rate * d_loss__dw_out
            model.bias_out -= learning_rate * d_loss__db_out
        
        #### Ведение журнала эпох ####
        train_mse, train_acc = compute_mse_and_acc(
            model, X_train, y_train)
        valid_mse, valid_acc = compute_mse_and_acc(
            model, X_valid, y_valid)
        
        train_acc, valid_acc = train_acc*100, valid_acc*100
        epoch_train_acc.append(train_acc)
        epoch_valid_acc.append(valid_acc)
        epoch_loss.append(train_mse)
        
        print(f'Эпоха: {e+1:03d}/{num_epochs:03d} '
              f'| Train MSE: {train_mse:.2f} '
              f'| Train Acc: {train_acc:.2f}% '
              f'| Valid Acc: {valid_acc:.2f}%')
    
    return epoch_loss, epoch_train_acc, epoch_valid_acc


if __name__ == "__main__":
    # Загрузка данных MNIST (используем функцию из предыдущего примера)
    import gzip
    import os
    from sklearn.model_selection import train_test_split
    
    def load_mnist_local(path):
        def load_images(filename):
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=16)
            return data.reshape(-1, 784)
        
        def load_labels(filename):
            with gzip.open(filename, 'rb') as f:
                data = np.frombuffer(f.read(), np.uint8, offset=8)
            return data
        
        X_train = load_images(os.path.join(path, 'train-images-idx3-ubyte.gz'))
        y_train = load_labels(os.path.join(path, 'train-labels-idx1-ubyte.gz'))
        X_test = load_images(os.path.join(path, 't10k-images-idx3-ubyte.gz'))
        y_test = load_labels(os.path.join(path, 't10k-labels-idx1-ubyte.gz'))
        
        X = np.concatenate([X_train, X_test], axis=0)
        y = np.concatenate([y_train, y_test], axis=0)
        return X, y
    
    print("Загрузка MNIST...")
    mnist_path = r'D:\GITHUB\PythonProjects\MNIST'
    X, y = load_mnist_local(mnist_path)
    
    # Масштабирование
    X = ((X / 255.) - .5) * 2
    
    # Разделение на train/valid/test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=10000, random_state=123, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=5000, random_state=123, stratify=y_temp)
    
    print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")
    
    # Создание модели
    print("\nСоздание модели MLP...")
    model = NeuralNetMLP(num_features=28*28,
                        num_hidden=50,
                        num_classes=10)
    
    # Проверка начальных значений
    _, probas = model.forward(X_valid)
    mse = mse_loss(y_valid, probas)
    predicted_labels = np.argmax(probas, axis=1)
    acc = accuracy(y_valid, predicted_labels)
    print(f'Начальная MSE при валидации: {mse:.2f}')
    print(f'Начальная точность при валидации: {acc*100:.2f}%')
    
    # Обучение модели
    print("\nОбучение модели...")
    np.random.seed(123)
    epoch_loss, epoch_train_acc, epoch_valid_acc = train(
        model, X_train, y_train, X_valid, y_valid,
        num_epochs=50, learning_rate=0.1)
    
    # Визуализация результатов обучения
    plt.plot(epoch_loss, label='Train MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.savefig('mlp_training_loss.png', dpi=300)
    plt.show()
    
    plt.plot(epoch_train_acc, label='Train Accuracy')
    plt.plot(epoch_valid_acc, label='Valid Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.savefig('mlp_training_accuracy.png', dpi=300)
    plt.show()
    
    # Оценка на тестовом наборе
    test_mse, test_acc = compute_mse_and_acc(model, X_test, y_test)
    print(f'\nТестовая MSE: {test_mse:.2f}')
    print(f'Тестовая точность: {test_acc*100:.2f}%')
