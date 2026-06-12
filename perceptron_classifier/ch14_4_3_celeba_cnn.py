"""
Раздел 14.4.3: Обучение классификатора улыбки на основе CNN
Классификация улыбающихся лиц с использованием CelebA датасета
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


def get_celeba_dataloaders(image_path='../celeba', batch_size=32, 
                          train_subset_size=16000, valid_subset_size=1000):
    """
    Создание DataLoader'ов для CelebA с аугментацией данных
    """
    # Функция для извлечения метки улыбки (атрибут 31)
    get_smile = lambda attr: attr[31]
    
    # Трансформации для обучения (с аугментацией)
    transform_train = transforms.Compose([
        transforms.RandomCrop([178, 178]),
        transforms.RandomHorizontalFlip(),
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
    ])
    
    # Трансформации для валидации и тестирования (без аугментации)
    transform = transforms.Compose([
        transforms.CenterCrop([178, 178]),
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
    ])
    
    # Загрузка наборов данных
    celeba_train_dataset = torchvision.datasets.CelebA(
        image_path, split='train',
        target_type='attr', download=False,
        transform=transform_train, target_transform=get_smile
    )
    
    celeba_valid_dataset = torchvision.datasets.CelebA(
        image_path, split='valid',
        target_type='attr', download=False,
        transform=transform, target_transform=get_smile
    )
    
    celeba_test_dataset = torchvision.datasets.CelebA(
        image_path, split='test',
        target_type='attr', download=False,
        transform=transform, target_transform=get_smile
    )
    
    print(f'Исходный размер обучающего набора: {len(celeba_train_dataset)}')
    print(f'Исходный размер валидационного набора: {len(celeba_valid_dataset)}')
    print(f'Исходный размер тестового набора: {len(celeba_test_dataset)}')
    
    # Создание подмножеств
    celeba_train_dataset = Subset(celeba_train_dataset, torch.arange(train_subset_size))
    celeba_valid_dataset = Subset(celeba_valid_dataset, torch.arange(valid_subset_size))
    
    print(f'\nРазмер подмножества для обучения: {len(celeba_train_dataset)}')
    print(f'Размер подмножества для валидации: {len(celeba_valid_dataset)}')
    
    # Создание загрузчиков данных
    torch.manual_seed(1)
    train_dl = DataLoader(celeba_train_dataset, batch_size, shuffle=True)
    valid_dl = DataLoader(celeba_valid_dataset, batch_size, shuffle=False)
    test_dl = DataLoader(celeba_test_dataset, batch_size, shuffle=False)
    
    return train_dl, valid_dl, test_dl


def create_cnn_model():
    """
    Создание CNN модели для классификации улыбки
    
    Архитектура:
    - 4 сверточных слоя (32, 64, 128, 256 каналов)
    - MaxPooling после первых 3 слоев
    - Dropout для регуляризации
    - Global Average Pooling
    - Полносвязный слой с сигмоидой
    """
    model = nn.Sequential()
    
    # Conv Block 1: 3 -> 32
    model.add_module('conv1', nn.Conv2d(
        in_channels=3, out_channels=32,
        kernel_size=3, padding=1
    ))
    model.add_module('relu1', nn.ReLU())
    model.add_module('pool1', nn.MaxPool2d(kernel_size=2))
    model.add_module('dropout1', nn.Dropout(p=0.5))
    
    # Conv Block 2: 32 -> 64
    model.add_module('conv2', nn.Conv2d(
        in_channels=32, out_channels=64,
        kernel_size=3, padding=1
    ))
    model.add_module('relu2', nn.ReLU())
    model.add_module('pool2', nn.MaxPool2d(kernel_size=2))
    model.add_module('dropout2', nn.Dropout(p=0.5))
    
    # Conv Block 3: 64 -> 128
    model.add_module('conv3', nn.Conv2d(
        in_channels=64, out_channels=128,
        kernel_size=3, padding=1
    ))
    model.add_module('relu3', nn.ReLU())
    model.add_module('pool3', nn.MaxPool2d(kernel_size=2))
    
    # Conv Block 4: 128 -> 256
    model.add_module('conv4', nn.Conv2d(
        in_channels=128, out_channels=256,
        kernel_size=3, padding=1
    ))
    model.add_module('relu4', nn.ReLU())
    
    # Global Average Pooling: 256 x 8 x 8 -> 256
    model.add_module('pool4', nn.AvgPool2d(kernel_size=8))
    model.add_module('flatten', nn.Flatten())
    
    # Полносвязный слой: 256 -> 1
    model.add_module('fc', nn.Linear(256, 1))
    model.add_module('sigmoid', nn.Sigmoid())
    
    return model


def train(model, num_epochs, train_dl, valid_dl, loss_fn, optimizer, device='cpu'):
    """
    Обучение модели
    
    Parameters:
    -----------
    model : nn.Module
        CNN модель
    num_epochs : int
        Количество эпох
    train_dl : DataLoader
        Загрузчик обучающих данных
    valid_dl : DataLoader
        Загрузчик валидационных данных
    loss_fn : nn.Module
        Функция потерь (BCELoss)
    optimizer : torch.optim.Optimizer
        Оптимизатор
    device : str
        Устройство ('cpu' или 'cuda')
        
    Returns:
    --------
    loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid
    """
    model = model.to(device)
    
    loss_hist_train = [0] * num_epochs
    accuracy_hist_train = [0] * num_epochs
    loss_hist_valid = [0] * num_epochs
    accuracy_hist_valid = [0] * num_epochs
    
    for epoch in range(num_epochs):
        # Обучение
        model.train()
        epoch_start = time.time()
        batch_iterator = tqdm(train_dl, desc=f'Эпоха {epoch+1}/{num_epochs}', 
                              total=len(train_dl), leave=False)
        for x_batch, y_batch in batch_iterator:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            pred = model(x_batch)[:, 0]
            loss = loss_fn(pred, y_batch.float())
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            loss_hist_train[epoch] += loss.item() * y_batch.size(0)
            is_correct = ((pred >= 0.5).float() == y_batch).float()
            accuracy_hist_train[epoch] += is_correct.sum().cpu().item()
            
            # Обновляем прогресс-бар
            processed = batch_iterator.n * y_batch.size(0)
            if processed > 0:
                current_loss = loss_hist_train[epoch] / processed
                current_acc = accuracy_hist_train[epoch] / processed
                batch_iterator.set_postfix({'loss': f'{current_loss:.4f}', 
                                            'acc': f'{current_acc:.4f}'})
        
        loss_hist_train[epoch] /= len(train_dl.dataset)
        accuracy_hist_train[epoch] /= len(train_dl.dataset)
        epoch_time = time.time() - epoch_start
        
        # Валидация
        model.eval()
        with torch.no_grad():
            for x_batch, y_batch in valid_dl:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                pred = model(x_batch)[:, 0]
                loss = loss_fn(pred, y_batch.float())
                
                loss_hist_valid[epoch] += loss.item() * y_batch.size(0)
                is_correct = ((pred >= 0.5).float() == y_batch).float()
                accuracy_hist_valid[epoch] += is_correct.sum().cpu().item()
        
        loss_hist_valid[epoch] /= len(valid_dl.dataset)
        accuracy_hist_valid[epoch] /= len(valid_dl.dataset)
        
        # Вывод прогресса
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f'Эпоха {epoch+1:2d} ({epoch_time:.1f}с): '
                  f'train_acc={accuracy_hist_train[epoch]:.4f}, '
                  f'val_acc={accuracy_hist_valid[epoch]:.4f}')
    
    return loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid


def plot_learning_curves(hist):
    """
    Визуализация кривых обучения
    """
    loss_hist_train, loss_hist_valid, accuracy_hist_train, accuracy_hist_valid = hist
    
    x_arr = np.arange(len(loss_hist_train)) + 1
    
    fig = plt.figure(figsize=(12, 4))
    
    # Потери
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(x_arr, loss_hist_train, '-o', label='Потери при обучении')
    ax.plot(x_arr, loss_hist_valid, '--<', label='Потери при валидации')
    ax.legend(fontsize=12)
    ax.set_xlabel('Эпоха', size=12)
    ax.set_ylabel('Потери', size=12)
    ax.set_title('Функция потерь BCE')
    
    # Точность
    ax = fig.add_subplot(1, 2, 2)
    ax.plot(x_arr, accuracy_hist_train, '-o', label='Точность при обучении')
    ax.plot(x_arr, accuracy_hist_valid, '--<', label='Точность при валидации')
    ax.legend(fontsize=12)
    ax.set_xlabel('Эпоха', size=12)
    ax.set_ylabel('Точность', size=12)
    ax.set_title('Accuracy')
    
    plt.tight_layout()
    plt.savefig('celeba_cnn_learning_curves.png', dpi=300, bbox_inches='tight')
    print("График сохранен как 'celeba_cnn_learning_curves.png'")
    plt.show()


def evaluate_test_set(model, test_dl, device='cpu'):
    """
    Оценка модели на тестовом наборе
    """
    model = model.to(device)
    model.eval()
    
    accuracy_test = 0
    with torch.no_grad():
        for x_batch, y_batch in test_dl:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            pred = model(x_batch)[:, 0]
            is_correct = ((pred >= 0.5).float() == y_batch).float()
            accuracy_test += is_correct.sum().cpu().item()
    
    accuracy_test /= len(test_dl.dataset)
    print(f'\nТочность при тестировании: {accuracy_test:.4f}')
    return accuracy_test


def visualize_predictions(model, test_dl, device='cpu', num_samples=10):
    """
    Визуализация предсказаний на тестовых примерах
    """
    model = model.to(device)
    model.eval()
    
    # Получаем один батч
    for x_batch, y_batch in test_dl:
        break
    
    x_batch = x_batch.to(device)
    
    with torch.no_grad():
        pred = model(x_batch)[:, 0] * 100  # В процентах
    
    fig = plt.figure(figsize=(15, 7))
    
    for j in range(num_samples):
        ax = fig.add_subplot(2, 5, j + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Переводим тензор в изображение (C, H, W) -> (H, W, C)
        img = x_batch[j].cpu().permute(1, 2, 0)
        ax.imshow(img)
        
        # Метка истинности
        if y_batch[j] == 1:
            label = 'Улыбка'
        else:
            label = 'Нет улыбки'
        
        # Текст с меткой и вероятностью
        ax.text(
            0.5, -0.15,
            f'GT: {label}\nPr(Smile)={pred[j]:.0f}%',
            size=12,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes
        )
    
    plt.tight_layout()
    plt.savefig('celeba_cnn_predictions.png', dpi=300, bbox_inches='tight')
    print("Предсказания сохранены как 'celeba_cnn_predictions.png'")
    plt.show()


def main():
    print("=" * 70)
    print("CNN для классификации улыбающихся лиц (CelebA)")
    print("=" * 70)
    
    # Определение устройства
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nИспользуемое устройство: {device}")
    
    # Загрузка данных
    print("\n" + "=" * 70)
    print("Загрузка данных CelebA")
    print("=" * 70)
    train_dl, valid_dl, test_dl = get_celeba_dataloaders(
        image_path='..',
        batch_size=32,
        train_subset_size=16000,
        valid_subset_size=1000
    )
    
    # Создание модели
    print("\n" + "=" * 70)
    print("Создание CNN модели")
    print("=" * 70)
    model = create_cnn_model()
    
    # Проверка размерностей
    print("\nПроверка размерностей:")
    x = torch.ones((4, 3, 64, 64))
    print(f"Вход: {x.shape}")
    
    # Последовательно проверяем размеры
    layers = ['conv1', 'relu1', 'pool1', 'conv2', 'relu2', 'pool2',
              'conv3', 'relu3', 'pool3', 'conv4', 'relu4', 'pool4', 'flatten', 'fc']
    
    temp_x = x
    for name in layers:
        layer = getattr(model, name)
        temp_x = layer(temp_x)
        if 'pool' in name or name == 'flatten' or name == 'fc':
            print(f"После {name:12}: {temp_x.shape}")
    
    # Подсчет параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nВсего параметров: {total_params:,}")
    print(f"Обучаемых параметров: {trainable_params:,}")
    
    # Функция потерь и оптимизатор
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Обучение
    print("\n" + "=" * 70)
    print("Обучение модели (30 эпох)")
    print("=" * 70)
    torch.manual_seed(1)
    num_epochs = 30
    
    hist = train(model, num_epochs, train_dl, valid_dl, loss_fn, optimizer, device)
    
    # Визуализация кривых обучения
    print("\n" + "=" * 70)
    print("Визуализация кривых обучения")
    print("=" * 70)
    plot_learning_curves(hist)
    
    # Оценка на тестовом наборе
    print("\n" + "=" * 70)
    print("Оценка на тестовом наборе")
    print("=" * 70)
    evaluate_test_set(model, test_dl, device)
    
    # Визуализация предсказаний
    print("\n" + "=" * 70)
    print("Визуализация предсказаний")
    print("=" * 70)
    visualize_predictions(model, test_dl, device, num_samples=10)
    
    # Сохранение модели
    torch.save(model.state_dict(), 'celeba_smile_cnn_model.pth')
    print("\n" + "=" * 70)
    print("Модель сохранена как 'celeba_smile_cnn_model.pth'")
    print("=" * 70)


if __name__ == '__main__':
    main()
