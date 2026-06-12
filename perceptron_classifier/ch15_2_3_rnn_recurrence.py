"""
Раздел 15.2.3: Скрытая и выходная рекуррентность
Демонстрация работы RNN с ручным вычислением прямого прохода
"""

import torch
import torch.nn as nn


def main():
    print("=" * 70)
    print("15.2.3. Скрытая и выходная рекуррентность в RNN")
    print("=" * 70)
    
    # Установка seed для воспроизводимости
    torch.manual_seed(1)
    
    # Создание RNN слоя
    rnn_layer = nn.RNN(
        input_size=5, 
        hidden_size=2,
        num_layers=1, 
        batch_first=True
    )
    
    # Получение весов и смещений
    w_xh = rnn_layer.weight_ih_l0
    w_hh = rnn_layer.weight_hh_l0
    b_xh = rnn_layer.bias_ih_l0
    b_hh = rnn_layer.bias_hh_l0
    
    print("\nПараметры RNN слоя:")
    print(f"Форма W_xh (вход -> скрытый): {w_xh.shape}")
    print(f"Форма W_hh (скрытый -> скрытый): {w_hh.shape}")
    print(f"Форма b_xh (смещение входа): {b_xh.shape}")
    print(f"Форма b_hh (смещение скрытого): {b_hh.shape}")
    
    # Входная последовательность длины 3
    x_seq = torch.tensor([[1.0]*5, [2.0]*5, [3.0]*5]).float()
    
    print(f"\nВходная последовательность (длина 3, размерность 5):")
    print(x_seq)
    
    # Прямой проход через RNN
    output, hn = rnn_layer(torch.reshape(x_seq, (1, 3, 5)))
    
    print(f"\nВыход RNN (форма: {output.shape}):")
    print(output)
    print(f"\nФинальное скрытое состояние (форма: {hn.shape}):")
    print(hn)
    
    # Ручное вычисление прямого прохода
    print("\n" + "=" * 70)
    print("Ручное вычисление прямого прохода")
    print("=" * 70)
    
    out_man = []
    
    for t in range(3):
        xt = torch.reshape(x_seq[t], (1, 5))
        print(f"\nШаг времени {t}:")
        print(f"  Вход: {xt.numpy()}")
        
        # Вычисление скрытого состояния: h_t = x_t @ W_xh^T + b_xh
        ht = torch.matmul(xt, torch.transpose(w_xh, 0, 1)) + b_xh
        print(f"  Скрытый (до рекуррентности): {ht.detach().numpy()}")
        
        # Добавление рекуррентной связи
        if t > 0:
            prev_h = out_man[t-1]
        else:
            prev_h = torch.zeros((ht.shape))
        
        # o_t = h_t + h_{t-1} @ W_hh^T + b_hh
        ot = ht + torch.matmul(prev_h, torch.transpose(w_hh, 0, 1)) + b_hh
        ot = torch.tanh(ot)
        
        out_man.append(ot)
        
        print(f"  Выход (ручной): {ot.detach().numpy()}")
        print(f"  Выход (RNN):    {output[:, t].detach().numpy()}")
        
        # Проверка совпадения
        if torch.allclose(ot, output[:, t], atol=1e-6):
            print(f"  ✓ Совпадает!")
        else:
            print(f"  ✗ Не совпадает!")
    
    print("\n" + "=" * 70)
    print("Объяснение архитектуры RNN")
    print("=" * 70)
    print("""
Типы рекуррентных связей:
1. Скрытый -> Скрытый (hidden-to-hidden, W_hh):
   - Стандартная RNN (используется в этом примере)
   - Предыдущее скрытое состояние влияет на текущее скрытое состояние
   
2. Выход -> Скрытый (output-to-hidden, W_oh):
   - Предыдущий выход добавляется к текущему скрытому состоянию
   - Полезно для задач, где выход важен для контекста
   
3. Выход -> Выход (output-to-output, W_oo):
   - Предыдущий выход влияет на текущий выход
   - Используется в авторегрессионных моделях

В этом примере используется hidden-to-hidden рекуррентность:
  h_t = tanh(x_t @ W_xh^T + b_xh + h_{t-1} @ W_hh^T + b_hh)
  
Где:
  - x_t: вход на шаге t
  - h_t: скрытое состояние на шаге t
  - W_xh: веса вход -> скрытый
  - W_hh: веса скрытый -> скрытый
  - b_xh, b_hh: смещения
    """)
    
    print("=" * 70)
    print("Демонстрация завершена!")
    print("=" * 70)


if __name__ == '__main__':
    main()
