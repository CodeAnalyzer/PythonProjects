"""
Пример использования обученной модели для предсказания улыбки на новых изображениях
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse


def create_cnn_model():
    """
    Создание CNN модели (та же архитектура, что при обучении)
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


def predict_smile(image_path, model_path='celeba_smile_cnn_model.pth', device='cpu'):
    """
    Предсказание улыбки на одном изображении
    
    Parameters:
    -----------
    image_path : str
        Путь к изображению
    model_path : str
        Путь к сохраненной модели
    device : str
        Устройство ('cpu' или 'cuda')
        
    Returns:
    --------
    smile_probability : float
        Вероятность улыбки (0-1)
    """
    # Загрузка модели
    model = create_cnn_model()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Трансформации (те же, что при обучении для valid/test)
    transform = transforms.Compose([
        transforms.CenterCrop([178, 178]),
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
    ])
    
    # Загрузка и подготовка изображения
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)  # Добавляем batch dimension
    
    # Предсказание
    with torch.no_grad():
        prediction = model(image_tensor)[:, 0].item()
    
    return prediction, image


def visualize_prediction(image_path, prediction):
    """
    Визуализация результата предсказания
    """
    image = Image.open(image_path).convert('RGB')
    
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    
    if prediction >= 0.5:
        label = 'Улыбка'
        color = 'green'
    else:
        label = 'Нет улыбки'
        color = 'red'
    
    plt.title(f'{label}\nВероятность: {prediction*100:.1f}%', 
              color=color, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('prediction_result.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='Предсказание улыбки на изображении')
    parser.add_argument('image_path', type=str, help='Путь к изображению')
    parser.add_argument('--model', type=str, default='celeba_smile_cnn_model.pth',
                        help='Путь к модели (по умолчанию: celeba_smile_cnn_model.pth)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Устройство: cpu, cuda или auto (по умолчанию)')
    args = parser.parse_args()
    
    print("=" * 70)
    print("Использование обученной модели для предсказания улыбки")
    print("=" * 70)
    
    # Определение устройства
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"\nИспользуемое устройство: {device}")
    
    # Путь к модели
    model_path = args.model
    print(f"Загрузка модели из: {model_path}")
    
    # Путь к изображению
    image_path = args.image_path
    print(f"Изображение: {image_path}")
    
    try:
        prediction, image = predict_smile(image_path, model_path, device)
        
        print(f"Вероятность улыбки: {prediction*100:.1f}%")
        
        if prediction >= 0.5:
            print("Результат: Улыбка 😊")
        else:
            print("Результат: Нет улыбки 😐")
        
        # Визуализация
        visualize_prediction(image_path, prediction)
        
    except FileNotFoundError:
        print(f"\nОшибка: изображение не найдено: {image_path}")
    except Exception as e:
        print(f"\nОшибка: {e}")


if __name__ == '__main__':
    main()
