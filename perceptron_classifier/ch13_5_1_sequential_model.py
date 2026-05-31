import torch
import torch.nn as nn

# ── Создание модели с помощью nn.Sequential ──
model = nn.Sequential(
    nn.Linear(4, 16),
    nn.ReLU(),
    nn.Linear(16, 32),
    nn.ReLU()
)

print('Архитектура модели:')
print(model)

# ── Инициализация весов первого слоя (Xavier/Glorot) ──
nn.init.xavier_uniform_(model[0].weight)
print('\nПервый слой (Linear(4, 16)) после инициализации Xavier:')
print(f'  Вес: mean={model[0].weight.mean().item():.4f}, std={model[0].weight.std().item():.4f}')
print(f'  Смещение: mean={model[0].bias.mean().item():.4f}')

# ── Вычисление L1-штрафа для весов второго слоя ──
ll_weight = 0.01
ll_penalty = ll_weight * model[2].weight.abs().sum()
print(f'\nL1-штраф для весов второго слоя (Linear(16, 32)):')
print(f'  L1-норма весов: {model[2].weight.abs().sum().item():.4f}')
print(f'  L1-штраф (коэффициент={ll_weight}): {ll_penalty.item():.4f}')

# ── Пример прямого прохода ──
x = torch.randn(1, 4)  # батч из 1 примера с 4 признаками
output = model(x)
print(f'\nПример прямого прохода:')
print(f'  Вход: {x.shape}')
print(f'  Выход: {output.shape}')
print(f'  Выходные значения: {output[0][:5].tolist()}...')  # первые 5 значений

# ── Настройка оптимизатора и функции потерь ──
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

print(f'\nНастройки обучения:')
print(f'  Оптимизатор: {optimizer.__class__.__name__}')
print(f'  Функция потерь: {loss_fn.__class__.__name__}')
print(f'  Скорость обучения: {learning_rate}')
