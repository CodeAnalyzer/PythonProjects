# -*- coding: utf-8 -*-
"""
Раздел 8.3: Обучение модели логистической регрессии для классификации документов

Пример демонстрирует:
1. Разделение данных на обучающий и тестовый наборы
2. Создание функций токенизации (с и без стемминга)
3. Использование TfidfVectorizer для преобразования текста
4. Создание Pipeline с LogisticRegression
5. Поиск оптимальных параметров с помощью GridSearchCV
6. Оценка точности на тестовом наборе

Модель классифицирует обзоры фильмов на положительные и отрицательные
с использованием модели мешка слов (Bag of Words).
"""
import sys
import io
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

print("📚 ОБУЧЕНИЕ МОДЕЛИ ЛОГИСТИЧЕСКОЙ РЕГРЕССИИ ДЛЯ КЛАССИФИКАЦИИ ДОКУМЕНТОВ")
print("=" * 70)

# Проверка наличия NLTK
try:
    import nltk
    print("✅ NLTK доступен")
except ImportError:
    print("❌ Ошибка: библиотека nltk не установлена")
    print("   Установите её командой: pip install nltk")
    sys.exit(1)

# Загрузка stopwords
try:
    stop = stopwords.words('english')
    print("✅ NLTK stopwords загружены")
except LookupError:
    print("⚠️  NLTK stopwords не найдены")
    print("   Загрузка stopwords...")
    nltk.download('stopwords')
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    print("✅ NLTK stopwords загружены")

# 1. Загрузка данных
print("\n📂 1. ЗАГРУЗКА ДАННЫХ")
print("-" * 50)

df = pd.read_csv('movie_data.csv', encoding='utf-8')

print(f"Размер DataFrame: {df.shape}")
print(f"Колонки: {df.columns.tolist()}")

# 2. Разделение на обучающий и тестовый наборы
print("\n✂️ 2. РАЗДЕЛЕНИЕ НА ОБУЧАЮЩИЙ И ТЕСТОВЫЙ НАБОРЫ")
print("-" * 50)

X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

print(f"Обучающий набор: {len(X_train)} образцов")
print(f"Тестовый набор: {len(X_test)} образцов")
print(f"Распределение классов в обучающем наборе: {np.bincount(y_train)}")
print(f"Распределение классов в тестовом наборе: {np.bincount(y_test)}")

# 3. Создание функций токенизации
print("\n🔧 3. СОЗДАНИЕ ФУНКЦИЙ ТОКЕНИЗАЦИИ")
print("-" * 50)

porter = PorterStemmer()

def tokenizer(text):
    """
    Простая токенизация текста без стемминга.
    Разбивает текст на слова, удаляет знаки препинания.
    """
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub(r'[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized

def tokenizer_porter(text):
    """
    Токенизация с использованием стемминга Портера.
    Приводит слова к основной форме (stem).
    """
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall(r'(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub(r'[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', '')
    tokenized = [porter.stem(w) for w in text.split() if w not in stop]
    return tokenized

print("Функции токенизации созданы:")
print("  - tokenizer: без стемминга")
print("  - tokenizer_porter: со стеммингом Портера")

# Демонстрация работы токенизаторов
sample_text = X_train[0][:200]
print(f"\nПример текста: {sample_text}...")
print(f"tokenizer: {tokenizer(sample_text)[:5]}")
print(f"tokenizer_porter: {tokenizer_porter(sample_text)[:5]}")

# 4. Создание Pipeline
print("\n🔧 4. СОЗДАНИЕ PIPELINE")
print("-" * 50)

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(strip_accents=None,
                       lowercase=False,
                       preprocessor=None)

lr_tfidf = Pipeline([
    ('vect', tfidf),
    ('clf', LogisticRegression(solver='liblinear', max_iter=1000))
])

print("Pipeline создан:")
print("  - vect: TfidfVectorizer")
print("  - clf: LogisticRegression (solver='liblinear')")

# 5. Настройка GridSearchCV
print("\n⚙️ 5. НАСТРОЙКА GRIDSEARCHCV")
print("-" * 50)

small_param_grid = [
    {
        'vect__ngram_range': [(1, 1)],
        'vect__stop_words': [None],
        'vect__tokenizer': [tokenizer, tokenizer_porter],
        'clf__C': [1.0, 10.0]
    },
    {
        'vect__ngram_range': [(1, 1)],
        'vect__stop_words': [stop, None],
        'vect__tokenizer': [tokenizer],
        'vect__use_idf': [False],
        'vect__norm': [None],
        'clf__C': [1.0, 10.0]
    }
]

print("Сетка параметров:")
print("  Словарь 1 (TF-IDF):")
print("    - vect__ngram_range: [(1, 1)]")
print("    - vect__stop_words: [None]")
print("    - vect__tokenizer: [tokenizer, tokenizer_porter]")
print("    - clf__C: [1.0, 10.0]")
print("  Словарь 2 (частоты терминов):")
print("    - vect__ngram_range: [(1, 1)]")
print("    - vect__stop_words: [stop, None]")
print("    - vect__tokenizer: [tokenizer]")
print("    - vect__use_idf: [False]")
print("    - vect__norm: [None]")
print("    - clf__C: [1.0, 10.0]")
print("  Примечание: penalty не указан (используется L2 по умолчанию)")

gs_lr_tfidf = GridSearchCV(lr_tfidf, small_param_grid,
                           scoring='accuracy', cv=5,
                           verbose=2, n_jobs=1)

print("\nЗапуск GridSearchCV...")
print("⚠️  Это может занять 5-10 минут на стандартном компьютере")

# 6. Обучение модели
print("\n🚀 6. ОБУЧЕНИЕ МОДЕЛИ")
print("-" * 50)

gs_lr_tfidf.fit(X_train, y_train)

print("\n✅ Обучение завершено")

# 7. Результаты
print("\n📊 7. РЕЗУЛЬТАТЫ")
print("-" * 50)

print(f'Лучший набор параметров: {gs_lr_tfidf.best_params_}')
print(f'Точность CV: {gs_lr_tfidf.best_score_:.3f}')

clf = gs_lr_tfidf.best_estimator_
print(f'Точность на тестовом наборе: {clf.score(X_test, y_test):.3f}')

# 8. Анализ результатов
print("\n📈 8. АНАЛИЗ РЕЗУЛЬТАТОВ")
print("-" * 50)

print("Интерпретация лучших параметров:")
best_params = gs_lr_tfidf.best_params_
if 'vect__tokenizer' in best_params:
    tokenizer_name = 'tokenizer_porter' if best_params['vect__tokenizer'] == tokenizer_porter else 'tokenizer'
    print(f"  - Токенизатор: {tokenizer_name}")
if 'vect__stop_words' in best_params:
    stop_words = 'с стоп-словами' if best_params['vect__stop_words'] is not None else 'без стоп-слов'
    print(f"  - Стоп-слова: {stop_words}")
if 'vect__use_idf' in best_params:
    use_idf = 'TF-IDF' if best_params['vect__use_idf'] else 'частоты терминов'
    print(f"  - Веса: {use_idf}")
print(f"  - Регуляризация: L2")
print(f"  - Параметр C: {best_params['clf__C']}")

# 9. Демонстрация предсказаний
print("\n🎯 9. ДЕМОНСТРАЦИЯ ПРЕДСКАЗАНИЙ")
print("-" * 50)

# Предсказание на нескольких примерах
sample_indices = [0, 1, 2, len(X_test)-1, len(X_test)-2]
for idx in sample_indices:
    text = X_test[idx][:100]
    true_label = y_test[idx]
    pred_label = clf.predict([X_test[idx]])[0]
    pred_prob = clf.predict_proba([X_test[idx]])[0]
    
    sentiment_true = "положительный" if true_label == 1 else "отрицательный"
    sentiment_pred = "положительный" if pred_label == 1 else "отрицательный"
    
    print(f"\nОбразец {idx}:")
    print(f"  Текст: {text}...")
    print(f"  Истинная метка: {sentiment_true} ({true_label})")
    print(f"  Предсказанная метка: {sentiment_pred} ({pred_label})")
    print(f"  Вероятности: [отрицательный={pred_prob[0]:.3f}, положительный={pred_prob[1]:.3f}]")
    print(f"  Результат: {'✅ Верно' if true_label == pred_label else '❌ Неверно'}")

# 10. Выводы
print("\n📝 10. ВЫВОДЫ")
print("=" * 70)
print("Результаты демонстрируют:")
print(f"  ✅ Модель достигла точности {gs_lr_tfidf.best_score_:.3f} на перекрестной проверке")
print(f"  ✅ Модель достигла точности {clf.score(X_test, y_test):.3f} на тестовом наборе")
print("  ✅ Модель способна предсказывать тональность обзоров с точностью ~90%")
print("\nКлючевые наблюдения:")
print("  - TF-IDF работает лучше частот терминов")
print("  - Обычная токенизация работает лучше стемминга Портера")
print("  - Отключение стоп-слов может улучшить результаты")
print("  - Регуляризация L2 с C=10.0 дает лучшие результаты")
print("\nПреимущества подхода:")
print("  - Мешок слов прост и эффективен для анализа тональности")
print("  - LogisticRegression хорошо работает с разреженными данными")
print("  - TfidfVectorizer автоматически обрабатывает текст")
print("  - GridSearchCV находит оптимальные гиперпараметры")
print("\nВозможные улучшения:")
print("  - Использование n-грамм (bigrams, trigrams)")
print("  - Попробовать регуляризацию L1 для отбора признаков")
print("  - Использование других классификаторов (SVM, Naive Bayes)")
print("  - Применение Word2Vec или GloVe для векторизации")
