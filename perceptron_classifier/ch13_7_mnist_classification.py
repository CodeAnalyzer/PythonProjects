import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# ── Шаг 1: Загрузка данных MNIST ──
image_path = './'
transform = transforms.Compose([
    transforms.ToTensor()
])

print('Загрузка MNIST...')
mnist_train_dataset = datasets.MNIST(
    root=image_path, train=True,
    transform=transform, download=True
)

mnist_test_dataset = datasets.MNIST(
    root=image_path, train=False,
    transform=transform, download=True
)

print(f'Обучающий набор: {len(mnist_train_dataset)} изображений')
print(f'Тестовый набор: {len(mnist_test_dataset)} изображений')

# ── DataLoader ──
batch_size = 64
torch.manual_seed(1)
train_dl = DataLoader(mnist_train_dataset, batch_size, shuffle=True)

# ── Шаг 2: Предобработка данных ──
# Преобразование ToTensor уже нормализует пиксели [0,255] -> [0,1]
# Метки - целые числа 0-9, преобразование не требуется

# ── Шаг 3: Построение модели ──
hidden_units = [32, 16]
image_size = mnist_train_dataset[0][0].shape
input_size = image_size[0] * image_size[1] * image_size[2]

print(f'\nРазмер изображения: {image_size}')
print(f'Размер входного вектора: {input_size}')

all_layers = [nn.Flatten()]

for hidden_unit in hidden_units:
    layer = nn.Linear(input_size, hidden_unit)
    all_layers.append(layer)
    all_layers.append(nn.ReLU())
    input_size = hidden_unit

all_layers.append(nn.Linear(hidden_units[-1], 10))
model = nn.Sequential(*all_layers)

print('\nАрхитектура модели:')
print(model)

# ── Шаг 4: Обучение модели ──
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

torch.manual_seed(1)
num_epochs = 20

print('\nОбучение модели...')
for epoch in range(num_epochs):
    accuracy_hist_train = 0
    for x_batch, y_batch in train_dl:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
        accuracy_hist_train += is_correct.sum()
    
    accuracy_hist_train /= len(train_dl.dataset)
    print(f'Эпоха {epoch:2d} Точность {accuracy_hist_train:.4f}')

# ── Оценка на тестовом наборе ──
print('\nОценка на тестовом наборе...')
pred = model(mnist_test_dataset.data / 255.)
is_correct = (torch.argmax(pred, dim=1) == mnist_test_dataset.targets).float()
print(f'Точность при тестировании: {is_correct.mean():.4f}')

# ── Демонстрация предсказаний для нескольких примеров ──
print('\nПримеры предсказаний:')
model.eval()
with torch.no_grad():
    for i in range(5):
        x, y = mnist_test_dataset[i]
        pred = model(x.unsqueeze(0))
        predicted = torch.argmax(pred, dim=1).item()
        print(f'  Пример {i}: предсказано={predicted}, реально={y}')
