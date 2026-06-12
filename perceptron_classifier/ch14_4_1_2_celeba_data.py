"""
Глава 14: Глубокое обучение с использованием сверточных нейронных сетей
Раздел 14.4: Классификация улыбающихся лиц с помощью CNN
Разделы 14.4.1-14.4.2: Загрузка набора данных CelebA и дополнение данных
"""

import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np


def visualize_transforms(celeba_train_dataset):
    """
    Визуализация различных трансформаций изображений:
    1. Обрезка до ограничительной рамки
    2. Отражение по горизонтали
    3. Изменение контраста
    4. Изменение яркости
    5. Обрезка по центру и изменение размера
    """
    fig = plt.figure(figsize=(16, 8.5))
    
    # Столбец 1: обрезка до ограничительной рамки
    ax = fig.add_subplot(2, 5, 1)
    img, attr = celeba_train_dataset[0]
    ax.set_title('Обрезка до\nограничительной\nрамки', size=15)
    ax.imshow(img)
    ax.axis('off')
    
    ax = fig.add_subplot(2, 5, 6)
    img_cropped = transforms.functional.crop(img, 50, 20, 128, 128)
    ax.imshow(img_cropped)
    ax.axis('off')
    
    # Столбец 2: отражение (по горизонтали)
    ax = fig.add_subplot(2, 5, 2)
    img, attr = celeba_train_dataset[1]
    ax.set_title('Отражение\n(по горизонтали)', size=15)
    ax.imshow(img)
    ax.axis('off')
    
    ax = fig.add_subplot(2, 5, 7)
    img_flipped = transforms.functional.hflip(img)
    ax.imshow(img_flipped)
    ax.axis('off')
    
    # Столбец 3: изменение контраста
    ax = fig.add_subplot(2, 5, 3)
    img, attr = celeba_train_dataset[2]
    ax.set_title('Изменение\nконтраста', size=15)
    ax.imshow(img)
    ax.axis('off')
    
    ax = fig.add_subplot(2, 5, 8)
    img_adj_contrast = transforms.functional.adjust_contrast(img, contrast_factor=2)
    ax.imshow(img_adj_contrast)
    ax.axis('off')
    
    # Столбец 4: изменение яркости
    ax = fig.add_subplot(2, 5, 4)
    img, attr = celeba_train_dataset[3]
    ax.set_title('Изменение\nяркости', size=15)
    ax.imshow(img)
    ax.axis('off')
    
    ax = fig.add_subplot(2, 5, 9)
    img_adj_brightness = transforms.functional.adjust_brightness(img, brightness_factor=1.3)
    ax.imshow(img_adj_brightness)
    ax.axis('off')
    
    # Столбец 5: обрезка относительно центра
    ax = fig.add_subplot(2, 5, 5)
    img, attr = celeba_train_dataset[4]
    ax.set_title('Обрезка по\nцентру и подгонка\nразмера', size=15)
    ax.imshow(img)
    ax.axis('off')
    
    ax = fig.add_subplot(2, 5, 10)
    img_center_crop = transforms.functional.center_crop(img, [int(0.7*218), int(0.7*178)])
    img_resized = transforms.functional.resize(img_center_crop, size=(218, 178))
    ax.imshow(img_resized)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('celeba_transforms.png', dpi=300, bbox_inches='tight')
    print("Визуализация трансформаций сохранена как 'celeba_transforms.png'")
    plt.show()


def visualize_augmentation_pipeline(celeba_train_dataset):
    """
    Визуализация конвейера аугментации данных:
    1. Случайная обрезка
    2. Случайное отражение
    3. Изменение размера
    """
    torch.manual_seed(1)
    fig = plt.figure(figsize=(14, 12))
    
    for i, (img, attr) in enumerate(celeba_train_dataset):
        if i >= 3:
            break
            
        # Оригинал
        ax = fig.add_subplot(3, 4, i*4+1)
        ax.imshow(img)
        if i == 0:
            ax.set_title('Оригинал', size=15)
        ax.axis('off')
        
        # Шаг 1: Случайная обрезка
        ax = fig.add_subplot(3, 4, i*4+2)
        img_transform = transforms.Compose([
            transforms.RandomCrop([178, 178])
        ])
        img_cropped = img_transform(img)
        ax.imshow(img_cropped)
        if i == 0:
            ax.set_title('Шаг 1:\nСлучайная обрезка', size=15)
        ax.axis('off')
        
        # Шаг 2: Случайное отражение
        ax = fig.add_subplot(3, 4, i*4+3)
        img_transform = transforms.Compose([
            transforms.RandomHorizontalFlip()
        ])
        img_flip = img_transform(img_cropped)
        ax.imshow(img_flip)
        if i == 0:
            ax.set_title('Шаг 2:\nСлучайное отражение', size=15)
        ax.axis('off')
        
        # Шаг 3: Изменение размера
        ax = fig.add_subplot(3, 4, i*4+4)
        img_resized = transforms.functional.resize(img_flip, size=(128, 128))
        ax.imshow(img_resized)
        if i == 0:
            ax.set_title('Шаг 3:\nИзменение размера', size=15)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('celeba_augmentation_pipeline.png', dpi=300, bbox_inches='tight')
    print("Конвейер аугментации сохранен как 'celeba_augmentation_pipeline.png'")
    plt.show()


def visualize_random_augmentations(celeba_train_dataset, num_epochs=5):
    """
    Визуализация случайных аугментаций на протяжении нескольких эпох
    """
    torch.manual_seed(1)
    
    # Функция для извлечения метки улыбки
    get_smile = lambda attr: attr[31]
    
    # Трансформация для обучения с аугментацией
    transform_train = transforms.Compose([
        transforms.RandomCrop([178, 178]),
        transforms.RandomHorizontalFlip(),
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
    ])
    
    celeba_train_dataset_transformed = torchvision.datasets.CelebA(
        '..', split='train',
        target_type='attr', download=False,
        transform=transform_train, target_transform=get_smile
    )
    
    data_loader = DataLoader(celeba_train_dataset_transformed, batch_size=2, shuffle=True)
    
    fig = plt.figure(figsize=(15, 6))
    
    for j in range(num_epochs):
        img_batch, label_batch = next(iter(data_loader))
        
        # Первое изображение в батче
        img = img_batch[0]
        ax = fig.add_subplot(2, 5, j + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f'Эпоха {j+1}', size=15)
        ax.imshow(img.permute(1, 2, 0))
        
        # Второе изображение в батче
        img = img_batch[1]
        ax = fig.add_subplot(2, 5, j + 6)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(img.permute(1, 2, 0))
    
    plt.tight_layout()
    plt.savefig('celeba_random_augmentations.png', dpi=300, bbox_inches='tight')
    print("Случайные аугментации сохранены как 'celeba_random_augmentations.png'")
    plt.show()


def get_celeba_dataloaders(image_path='./', batch_size=32, 
                          train_subset_size=16000, valid_subset_size=1000):
    """
    Создание DataLoader'ов для CelebA с аугментацией данных
    
    Parameters:
    -----------
    image_path : str
        Путь к данным CelebA
    batch_size : int
        Размер пакета
    train_subset_size : int
        Размер подмножества для обучения
    valid_subset_size : int
        Размер подмножества для валидации
        
    Returns:
    --------
    train_dl, valid_dl, test_dl : DataLoader
        Загрузчики данных для обучения, валидации и тестирования
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


def main():
    print("=" * 70)
    print("Подготовка данных CelebA для классификации улыбающихся лиц")
    print("=" * 70)
    
    image_path = '..'  # Корень, где находится папка celeba/ (base_folder для torchvision)
    
    # Загрузка полного набора для визуализации (без трансформаций)
    print("\nЗагрузка данных CelebA для визуализации...")
    print("Это может занять несколько минут (чтение 162770 изображений)...")
    import time
    start_time = time.time()
    celeba_train_dataset_full = torchvision.datasets.CelebA(
        image_path, split='train',
        target_type='attr', download=False
    )
    elapsed = time.time() - start_time
    print(f'Обучающий набор загружен: {len(celeba_train_dataset_full)} примеров (за {elapsed:.1f} сек)')
    
    # Проверка первого примера
    img, attr = celeba_train_dataset_full[0]
    print(f'Размер изображения: {img.size}')
    print(f'Количество атрибутов: {len(attr)}')
    print(f'Атрибут улыбки (31-й): {attr[31].item()}')
    
    # Визуализация трансформаций
    print("\nВизуализация различных трансформаций...")
    visualize_transforms(celeba_train_dataset_full)
    
    # Визуализация конвейера аугментации
    print("\nВизуализация конвейера аугментации...")
    visualize_augmentation_pipeline(celeba_train_dataset_full)
    
    # Визуализация случайных аугментаций
    print("\nВизуализация случайных аугментаций...")
    visualize_random_augmentations(celeba_train_dataset_full, num_epochs=5)
    
    # Создание DataLoader'ов
    print("\n" + "=" * 70)
    print("Создание DataLoader'ов")
    print("=" * 70)
    train_dl, valid_dl, test_dl = get_celeba_dataloaders(
        image_path=image_path,
        batch_size=32,
        train_subset_size=16000,
        valid_subset_size=1000
    )
    
    # Проверка размеров батчей
    print("\nПроверка загрузчиков данных:")
    for x_batch, y_batch in train_dl:
        print(f'Батч обучения: X={x_batch.shape}, y={y_batch.shape}')
        break
    
    for x_batch, y_batch in valid_dl:
        print(f'Батч валидации: X={x_batch.shape}, y={y_batch.shape}')
        break
    
    for x_batch, y_batch in test_dl:
        print(f'Батч тестирования: X={x_batch.shape}, y={y_batch.shape}')
        break
    
    print("\n" + "=" * 70)
    print("Подготовка данных завершена!")
    print("Готово к обучению CNN в разделе 14.4.3")
    print("=" * 70)


if __name__ == "__main__":
    main()
