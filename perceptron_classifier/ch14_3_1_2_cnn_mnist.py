"""
Глава 14: Глубокое обучение с использованием сверточных нейронных сетей
Разделы 14.3.1-14.3.2: Архитектура многослойной CNN и загрузка данных
Реализация CNN для классификации MNIST
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset


def load_mnist_data(image_path='./', batch_size=64):
    """
    Загрузка и разделение набора данных MNIST
    
    Parameters:
    -----------
    image_path : str
        Путь для сохранения/загрузки данных
    batch_size : int
        Размер пакета
        
    Returns:
    --------
    train_dl, valid_dl, test_dl : DataLoader
        Загрузчики данных для обучения, валидации и тестирования
    """
    # Преобразование данных в тензоры
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    # Загрузка обучающего набора
    mnist_dataset = torchvision.datasets.MNIST(
        root=image_path, 
        train=True,
        transform=transform, 
        download=False
    )
    
    # Разделение на валидационный (первые 10000) и обучающий наборы
    mnist_valid_dataset = Subset(mnist_dataset, torch.arange(10000))
    mnist_train_dataset = Subset(mnist_dataset, torch.arange(10000, len(mnist_dataset)))
    
    # Загрузка тестового набора
    mnist_test_dataset = torchvision.datasets.MNIST(
        root=image_path, 
        train=False,
        transform=transform, 
        download=False
    )
    
    # Создание загрузчиков данных
    torch.manual_seed(1)
    train_dl = DataLoader(mnist_train_dataset, batch_size, shuffle=True)
    valid_dl = DataLoader(mnist_valid_dataset, batch_size, shuffle=False)
    test_dl = DataLoader(mnist_test_dataset, batch_size, shuffle=False)
    
    return train_dl, valid_dl, test_dl, mnist_test_dataset


def create_cnn_model():
    """
    Создание CNN модели с архитектурой:
    - Conv1: 1 -> 32 каналов, ядро 5x5, padding=2
    - ReLU
    - MaxPool2d: 2x2
    - Conv2: 32 -> 64 каналов, ядро 5x5, padding=2
    - ReLU
    - MaxPool2d: 2x2
    - Flatten
    - FC1: 3136 -> 1024
    - ReLU
    - Dropout: 0.5
    - FC2: 1024 -> 10
    
    Returns:
    --------
    model : nn.Sequential
        Модель CNN
    """
    model = nn.Sequential()
    
    # Первый сверточный блок
    model.add_module('conv1', nn.Conv2d(
        in_channels=1, out_channels=32,
        kernel_size=5, padding=2
    ))
    model.add_module('relu1', nn.ReLU())
    model.add_module('pool1', nn.MaxPool2d(kernel_size=2))
    
    # Второй сверточный блок
    model.add_module('conv2', nn.Conv2d(
        in_channels=32, out_channels=64,
        kernel_size=5, padding=2
    ))
    model.add_module('relu2', nn.ReLU())
    model.add_module('pool2', nn.MaxPool2d(kernel_size=2))
    
    # Выравнивание для полносвязных слоев
    model.add_module('flatten', nn.Flatten())
    
    # Полносвязные слои
    model.add_module('fc1', nn.Linear(3136, 1024))
    model.add_module('relu3', nn.ReLU())
    model.add_module('dropout', nn.Dropout(p=0.5))
    model.add_module('fc2', nn.Linear(1024, 10))
    
    return model


def train(model, num_epochs, train_dl, valid_dl, loss_fn, optimizer, device='cpu'):
    """
    Обучение модели CNN
    
    Parameters:
    -----------
    model : nn.Module
        Модель для обучения
    num_epochs : int
        Количество эпох
    train_dl : DataLoader
        Загрузчик обучающих данных
    valid_dl : DataLoader
        Загрузчик валидационных данных
    loss_fn : callable
        Функция потерь
    optimizer : torch.optim.Optimizer
        Оптимизатор
    device : str
        Устройство ('cpu' или 'cuda')
        
    Returns:
    --------
    loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid
        История потерь и точности для обучения и валидации
    """
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        for x_batch, y_batch in train_dl:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            pred = model(x_batch)
            loss = loss_fn(pred, y_batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            loss_hist_train[epoch] += loss.item() * y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum()
        
        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)
        
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                pred = model(x_batch)
                loss = loss_fn(pred, y_batch)
                loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
                accuracy_hist_valid[epoch] += is_correct.sum()
        
        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)
        
        print(f'Эпоха {epoch+1}/{num_epochs}: '
              f'Train Acc: {accuracy_hist_train[epoch]:.4f} '
              f'Val Acc: {accuracy_hist_valid[epoch]:.4f}')
    
    # Конвертируем тензоры в CPU для визуализации
    return [x if isinstance(x, float) else x.cpu().item() if isinstance(x, torch.Tensor) else x 
            for x in loss_hist_train], \
           [x if isinstance(x, float) else x.cpu().item() if isinstance(x, torch.Tensor) else x 
            for x in loss_hist_valid], \
           [x if isinstance(x, float) else x.cpu().item() if isinstance(x, torch.Tensor) else x 
            for x in accuracy_hist_train], \
           [x if isinstance(x, float) else x.cpu().item() if isinstance(x, torch.Tensor) else x 
            for x in accuracy_hist_valid]


def plot_learning_curves(hist):
    """
    Визуализация кривых обучения
    
    Parameters:
    -----------
    hist : tuple
        Кортеж с историей (loss_train, loss_valid, acc_train, acc_valid)
    """
    loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid = hist
    
    x_arr = np.arange(len(loss_hist_train)) + 1
    
    fig = plt.figure(figsize=(12, 4))
    
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x_arr, loss_hist_train, '-o', label='Потери при обучении')
    ax.plot(x_arr, loss_hist_valid, '--<', label='Потери при валидации')
    ax.legend(fontsize=15)
    ax.set_xlabel('Эпоха', size=15)
    ax.set_ylabel('Потери', size=15)
    
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x_arr, accuracy_hist_train, '-o', label='Точность при обучении')
    ax.plot(x_arr, accuracy_hist_valid, '--<', label='Точность при валидации')
    ax.legend(fontsize=15)
    ax.set_xlabel('Эпоха', size=15)
    ax.set_ylabel('Точность', size=15)
    
    plt.tight_layout()
    plt.savefig('cnn_learning_curves.png', dpi=300, bbox_inches='tight')
    print("График сохранен как 'cnn_learning_curves.png'")
    plt.show()


def evaluate_test_set(model, test_dataset, device='cpu'):
    """
    Оценка модели на тестовом наборе
    
    Parameters:
    -----------
    model : nn.Module
        Обученная модель
    test_dataset : torchvision.datasets.MNIST
        Тестовый набор данных
    device : str
        Устройство ('cpu' или 'cuda')
    """
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        pred = model(test_dataset.data.unsqueeze(1).float().to(device) / 255.)
        is_correct = (torch.argmax(pred, dim=1) == test_dataset.targets.to(device)).float()
        print(f'Точность при тестировании: {is_correct.mean().cpu().item():.4f}')


def visualize_predictions(model, test_dataset, num_samples=12, device='cpu'):
    """
    Визуализация предсказаний модели
    
    Parameters:
    -----------
    model : nn.Module
        Обученная модель
    test_dataset : torchvision.datasets.MNIST
        Тестовый набор данных
    num_samples : int
        Количество примеров для визуализации
    device : str
        Устройство ('cpu' или 'cuda')
    """
    model = model.to(device)
    model.eval()
    
    fig = plt.figure(figsize=(12, 4))
    
    for i in range(num_samples):
        ax = fig.add_subplot(2, 6, i + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        img = test_dataset[i][0][0, :, :]
        pred = model(img.unsqueeze(0).unsqueeze(1).float().to(device))
        y_pred = torch.argmax(pred).cpu()
        
        ax.imshow(img, cmap='gray_r')
        ax.text(0.9, 0.1, y_pred.item(),
                size=15, color='blue',
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig('cnn_predictions.png', dpi=300, bbox_inches='tight')
    print("Предсказания сохранены как 'cnn_predictions.png'")
    plt.show()


def main():
    print("=" * 70)
    print("CNN для классификации MNIST")
    print("=" * 70)
    
    # Определение устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nИспользуемое устройство: {device}")
    
    # Загрузка данных
    print("\nЗагрузка данных MNIST...")
    train_dl, valid_dl, test_dl, mnist_test_dataset = load_mnist_data(
        image_path='./', 
        batch_size=64
    )
    print(f"Обучающий набор: {len(train_dl.dataset)} примеров")
    print(f"Валидационный набор: {len(valid_dl.dataset)} примеров")
    print(f"Тестовый набор: {len(test_dl.dataset)} примеров")
    
    # Создание модели
    print("\nСоздание CNN модели...")
    model = create_cnn_model()
    
    # Проверка размерностей
    print("\nПроверка размерностей:")
    x = torch.ones((4, 1, 28, 28))
    print(f"Вход: {x.shape}")
    x = model.conv1(x)
    print(f"После conv1: {x.shape}")
    x = model.relu1(x)
    x = model.pool1(x)
    print(f"После pool1: {x.shape}")
    x = model.conv2(x)
    print(f"После conv2: {x.shape}")
    x = model.relu2(x)
    x = model.pool2(x)
    print(f"После pool2: {x.shape}")
    x = model.flatten(x)
    print(f"После flatten: {x.shape}")
    x = model.fc1(x)
    print(f"После fc1: {x.shape}")
    x = model.fc2(x)
    print(f"После fc2: {x.shape}")
    
    # Количество параметров
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nОбщее количество параметров: {total_params:,}")
    
    # Функция потерь и оптимизатор
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Обучение
    print("\n" + "=" * 70)
    print("Обучение модели")
    print("=" * 70)
    torch.manual_seed(1)
    num_epochs = 20
    hist = train(model, num_epochs, train_dl, valid_dl, loss_fn, optimizer, device)
    
    # Визуализация кривых обучения
    print("\nВизуализация кривых обучения...")
    plot_learning_curves(hist)
    
    # Оценка на тестовом наборе
    print("\n" + "=" * 70)
    print("Оценка на тестовом наборе")
    print("=" * 70)
    evaluate_test_set(model, mnist_test_dataset, device)
    
    # Визуализация предсказаний
    print("\nВизуализация предсказаний...")
    visualize_predictions(model, mnist_test_dataset, num_samples=12, device=device)
    
    # Сохранение модели
    torch.save(model.state_dict(), 'cnn_mnist_model.pth')
    print("\nМодель сохранена как 'cnn_mnist_model.pth'")


if __name__ == "__main__":
    main()
