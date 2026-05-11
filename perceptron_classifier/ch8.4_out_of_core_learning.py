# -*- coding: utf-8 -*-
"""
Раздел 8.4: Работа с большими данными: онлайн-алгоритмы и внешнее обучение

Пример демонстрирует:
1. Использование out-of-core learning для данных, не помещающихся в память
2. Функцию stream_docs для потокового чтения документов
3. Функцию get_minibatch для получения мини-пакетов
4. HashingVectorizer вместо CountVectorizer/TfidfVectorizer
5. SGDClassifier с partial_fit для онлайн-обучения
6. Обучение на мини-пакетах без загрузки всех данных в память

Out-of-core learning позволяет работать с большими наборами данных,
постепенно обучая классификатор на меньших фрагментах.
"""
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier

print("📊 РАБОТА С БОЛЬШИМИ ДАННЫМИ: ОНЛАЙН-АЛГОРИТМЫ И ВНЕШНЕЕ ОБУЧЕНИЕ")
print("=" * 70)

# Проверка наличия NLTK stopwords
try:
    stop = stopwords.words('english')
    print("✅ NLTK stopwords загружены")
except LookupError:
    print("⚠️  NLTK stopwords не найдены")
    print("   Загрузка stopwords...")
    import nltk
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    print("✅ NLTK stopwords загружены")

# 1. Определение функции tokenizer
print("\n🔧 1. ФУНКЦИЯ ТОКЕНИЗАЦИИ")
print("-" * 50)

def tokenizer(text):
    """
    Очищает текст и разбивает на токены.
    Удаляет HTML-теги, эмодзи, стоп-слова.
    """
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub(r'[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

print("Функция tokenizer определена")
print("  - Удаляет HTML-теги")
print("  - Извлекает эмодзи")
print("  - Удаляет стоп-слова")

# 2. Определение функции stream_docs
print("\n📖 2. ФУНКЦИЯ ПОТОКОВОГО ЧТЕНИЯ ДОКУМЕНТОВ")
print("-" * 50)

def stream_docs(path):
    """
    Генератор, который читает документы по одному из CSV-файла.
    Возвращает кортеж (текст, метка) для каждого документа.
    """
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # пропуск заголовка
        for line in csv:
            text, label = line[:-3], int(line[-2])
            yield text, label

print("Функция stream_docs определена")
print("  - Читает документы по одному")
print("  - Не загружает весь файл в память")

# 3. Проверка работы stream_docs
print("\n✅ 3. ПРОВЕРКА РАБОТЫ STREAM_DOCS")
print("-" * 50)

doc_stream = stream_docs(path='movie_data.csv')
first_doc = next(doc_stream)
print(f"Первый документ:")
print(f"  Текст: {first_doc[0][:100]}...")
print(f"  Метка: {first_doc[1]}")

# 4. Определение функции get_minibatch
print("\n📦 4. ФУНКЦИЯ ПОЛУЧЕНИЯ МИНИ-ПАКЕТОВ")
print("-" * 50)

def get_minibatch(doc_stream, size):
    """
    Получает заданное количество документов из потока.
    Возвращает кортеж (список документов, список меток).
    """
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y

print("Функция get_minibatch определена")
print("  - Получает size документов из потока")
print("  - Возвращает None, если документы закончились")

# 5. Инициализация HashingVectorizer и SGDClassifier
print("\n🔧 5. ИНИЦИАЛИЗАЦИЯ ВЕКТОРИЗАТОРА И КЛАССИФИКАТОРА")
print("-" * 50)

vect = HashingVectorizer(decode_error='ignore',
                       n_features=2**21,
                       preprocessor=None,
                       tokenizer=tokenizer)

clf = SGDClassifier(loss='log_loss', random_state=1)

print("HashingVectorizer инициализирован:")
print("  - n_features=2**21 (2,097,152 признаков)")
print("  - Использует хеширование (murmurHash3)")
print("  - Не хранит словарь в памяти")
print("\nSGDClassifier инициализирован:")
print("  - loss='log_loss' (логистическая регрессия)")
print("  - random_state=1")
print("  - Поддерживает partial_fit для онлайн-обучения")

# 6. Обучение на внешних данных
print("\n🚀 6. ОБУЧЕНИЕ НА ВНЕШНИХ ДАННЫХ")
print("-" * 50)

# Проверка наличия pyprind
try:
    import pyprind
    use_pyprind = True
    print("✅ PyPrind доступен")
except ImportError:
    use_pyprind = False
    print("⚠️  PyPrind не установлен, будет использоваться встроенный прогресс")

# Сброс потока документов
doc_stream = stream_docs(path='movie_data.csv')

classes = np.array([0, 1])
n_minibatches = 45
minibatch_size = 1000

print(f"\nОбучение на {n_minibatches} мини-пакетах по {minibatch_size} документов")

if use_pyprind:
    pbar = pyprind.ProgBar(n_minibatches, stream=sys.stdout)
else:
    print(f"Обработка {n_minibatches} мини-пакетов...")

for _ in range(n_minibatches):
    X_train, y_train = get_minibatch(doc_stream, size=minibatch_size)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    
    if use_pyprind:
        pbar.update()
    elif (_ + 1) % 5 == 0:
        print(f"  Обработано: {_ + 1}/{n_minibatches} мини-пакетов")

print("\n✅ Обучение завершено")

# 7. Оценка точности на тестовом наборе
print("\n📊 7. ОЦЕНКА ТОЧНОСТИ НА ТЕСТОВОМ НАБОРЕ")
print("-" * 50)

X_test, y_test = get_minibatch(doc_stream, size=5000)
if X_test is not None:
    X_test = vect.transform(X_test)
    accuracy = clf.score(X_test, y_test)
    print(f'Точность: {accuracy:.3f}')
else:
    print("⚠️  Недостаточно документов для тестирования")

# 8. Обновление модели на тестовых данных
print("\n🔄 8. ОБНОВЛЕНИЕ МОДЕЛИ НА ТЕСТОВЫХ ДАННЫХ")
print("-" * 50)

if X_test is not None:
    clf = clf.partial_fit(X_test, y_test)
    print("✅ Модель обновлена на тестовых данных")
    print("   Это позволяет улучшить модель с помощью новых данных")

# 9. Сравнение с предыдущим методом
print("\n📈 9. СРАВНЕНИЕ С ПРЕДЫДУЩИМ МЕТОДОМ")
print("-" * 50)

print("Сравнение методов:")
print("  Предыдущий метод (GridSearchCV + LogisticRegression):")
print("    - Точность: ~0.899")
print("    - Время: 5-10 минут")
print("    - Память: загружает все данные в RAM")
print("\n  Текущий метод (Out-of-core learning):")
if X_test is not None:
    print(f"    - Точность: {accuracy:.3f}")
print("    - Время: <1 минуты")
print("    - Память: не загружает все данные в RAM")

# 10. Выводы
print("\n📝 10. ВЫВОДЫ")
print("=" * 70)
print("Результаты демонстрируют:")
if X_test is not None:
    print(f"  ✅ Модель достигла точности {accuracy:.3f}")
print("  ✅ Out-of-core learning эффективно использует память")
print("  ✅ Обучение заняло менее минуты")
print("\nПреимущества out-of-core learning:")
print("  - Работает с данными, не помещающимися в память")
print("  - Постепенное обучение на мини-пакетах")
print("  - Низкое потребление RAM")
print("  - Быстрое обучение")
print("\nКлючевые компоненты:")
print("  - HashingVectorizer: хеширование вместо хранения словаря")
print("  - SGDClassifier: стохастический градиентный спуск")
print("  - partial_fit: обучение на мини-пакетах")
print("  - stream_docs: потоковое чтение документов")
print("\nКомпромиссы:")
print("  - Точность немного ниже (~87% vs ~90%)")
print("  - Невозможно восстановить исходные признаки из хешей")
print("  - Возможны коллизии хешей (уменьшаются при большом n_features)")
print("\nКогда использовать out-of-core learning:")
print("  - Данные не помещаются в память")
print("  - Нужно быстро обучить модель")
print("  - Данные поступают в реальном времени")
print("  - Ограничены вычислительные ресурсы")
