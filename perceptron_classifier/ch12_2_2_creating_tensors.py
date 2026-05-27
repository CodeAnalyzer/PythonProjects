import torch
import numpy as np

np.set_printoptions(precision=3)

a = [1, 2, 3]
b = np.array([4, 5, 6], dtype=np.int32)

t_a = torch.tensor(a)
t_b = torch.from_numpy(b)

print('Тензор из списка:', t_a)
print('Тензор из массива NumPy:', t_b)

t_ones = torch.ones(2, 3)
print('Форма тензора из единиц:', t_ones.shape)
print('Тензор из единиц:', t_ones)

rand_tensor = torch.rand(2, 3)
print('Тензор случайных значений:', rand_tensor)

# ── 12.2.3. Управление типом данных и формой тензора ──

t_a_new = t_a.to(torch.int64)
print('Тип данных после преобразования:', t_a_new.dtype)

# транспонирование тензора
t = torch.rand(3, 5)
t_tr = torch.transpose(t, 0, 1)
print('Транспонирование:', t.shape, ' --> ', t_tr.shape)

# изменение формы тензора
t = torch.zeros(30)
t_reshape = t.reshape(5, 6)
print('Изменение формы (reshape):', t_reshape.shape)

# удаление ненужных измерений
t = torch.zeros(1, 2, 1, 4, 1)
t_sqz = torch.squeeze(t, 2)
print('Удаление измерения (squeeze):', t.shape, ' --> ', t_sqz.shape)

# ── 12.2.4. Применение математических операций ──

torch.manual_seed(1)
t1 = 2 * torch.rand(5, 2) - 1
t2 = torch.normal(mean=0, std=1, size=(5, 2))

# поэлементное произведение
t3 = torch.multiply(t1, t2)
print('Поэлементное произведение t1 * t2:', t3)

# среднее значение каждого столбца
t4 = torch.mean(t1, axis=0)
print('Среднее по столбцам:', t4)

# матричное произведение t1 @ t2^T
t5 = torch.matmul(t1, torch.transpose(t2, 0, 1))
print('Матричное произведение t1 @ t2^T:', t5)

# матричное произведение t1^T @ t2
t6 = torch.matmul(torch.transpose(t1, 0, 1), t2)
print('Матричное произведение t1^T @ t2:', t6)

# L2-норма
norm_t1 = torch.linalg.norm(t1, ord=2, dim=1)
print('L2-норма тензора t1:', norm_t1)

# проверка через NumPy
print('Проверка L2-нормы через NumPy:', np.sqrt(np.sum(np.square(t1.numpy()), axis=1)))

# ── 12.2.5. Разделение, стекирование и конкатенация ──

# разделение на заданное количество фрагментов
torch.manual_seed(1)
t = torch.rand(6)
print('Тензор для разделения (chunk):', t)
t_splits = torch.chunk(t, 3)
print('Результат chunk на 3 части:', [item.numpy() for item in t_splits])

# разделение по заданным размерам фрагментов
torch.manual_seed(1)
t = torch.rand(5)
print('Тензор для разделения (split):', t)
t_splits = torch.split(t, split_size_or_sections=[3, 2])
print('Результат split на [3, 2]:', [item.numpy() for item in t_splits])

# конкатенация
A = torch.ones(3)
B = torch.zeros(2)
C = torch.cat([A, B], axis=0)
print('Конкатенация A и B:', C)

# стекирование
A = torch.ones(3)
B = torch.zeros(3)
S = torch.stack([A, B], axis=1)
print('Стекирование A и B:', S)
