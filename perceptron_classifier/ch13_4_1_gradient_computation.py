import torch

# ── Определение параметров модели с requires_grad=True ──
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)

# Входные данные и целевое значение
x = torch.tensor([1.4])
y = torch.tensor([2.1])

# ── Прямой проход (forward pass) ──
# z = w * x + b
z = torch.add(torch.mul(w, x), b)

# ── Вычисление функции потерь (MSE) ──
# loss = (y - z)^2
loss = (y - z).pow(2).sum()

print(f'Прямой проход:')
print(f'  w = {w.item():.2f}, b = {b.item():.2f}')
print(f'  x = {x.item():.2f}, y = {y.item():.2f}')
print(f'  z = wx + b = {z.item():.4f}')
print(f'  loss = (y - z)^2 = {loss.item():.4f}')

# ── Обратный проход (backward pass) ──
# Вычисление градиентов
loss.backward()

print(f'\nГрадиенты после loss.backward():')
print(f'  dL/dw: {w.grad.item():.4f}')
print(f'  dL/db:  {b.grad.item():.4f}')

# ── Проверка градиентов вручную по формуле ──
# dL/dw = 2 * x * (w*x + b - y)
# dL/db = 2 * (w*x + b - y)
print(f'\nПроверка градиентов вручную:')
dL_dw_manual = 2 * x * ((w * x + b) - y)
dL_db_manual = 2 * ((w * x + b) - y)
print(f'  dL/dw (ручной расчет): {dL_dw_manual.item():.4f}')
print(f'  dL/db (ручной расчет):  {dL_db_manual.item():.4f}')

print(f'\nСовпадение градиентов:')
print(f'  dL/dw: {torch.isclose(w.grad, dL_dw_manual)}')
print(f'  dL/db:  {torch.isclose(b.grad, dL_db_manual)}')
