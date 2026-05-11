"""
Реализация LDA (Latent Dirichlet Allocation) в библиотеке scikit-learn
Глава 8.5.2 - Моделирование тем с использованием скрытого распределения Дирихле
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Проверка наличия pyprind для отображения прогресса
try:
    import pyprind
    use_pyprind = True
    print("✅ PyPrind доступен для отображения прогресса")
except ImportError:
    use_pyprind = False
    print("⚠️  PyPrind не установлен, будет использоваться стандартный вывод")

# Загрузка набора обзоров фильмов в DataFrame pandas из локального файла
print("\n📂 Загрузка данных из movie_data.csv...")
if use_pyprind:
    pbar = pyprind.ProgBar(1, title='Загрузка данных')
df = pd.read_csv('movie_data.csv', encoding='utf-8')
if use_pyprind:
    pbar.update()

# Следующая строка нужна для некоторых компьютеров:
df = df.rename(columns={"0": "review", "1": "sentiment"})

print("✅ Загружено обзоров:", df.shape[0])
print("\nПервые несколько строк данных:")
print(df.head())

# Применяем CountVectorizer для создания матрицы набора слов
# Используем встроенную библиотеку стоп-слов английского языка scikit-learn
print("\n🔄 Векторизация текста с CountVectorizer...")
if use_pyprind:
    pbar = pyprind.ProgBar(1, title='Векторизация')

count = CountVectorizer(stop_words='english',
                       max_df=.1,
                       max_features=5000)

X = count.fit_transform(df['review'].values)

if use_pyprind:
    pbar.update()

print(f"✅ Размерность матрицы мешка слов: {X.shape}")

# Обучение оценщика LatentDirichletAllocation на матрице набора слов
# Выводим 10 различных тем из документов
# learning_method='batch' - обучение на всех данных за одну итерацию (медленнее, но точнее)
lda = LatentDirichletAllocation(n_components=10,
                                random_state=123,
                                learning_method='batch',
                                max_iter=10,
                                verbose=1)

print("\n🚀 Обучение LDA модели (learning_method='batch')")
print("⏳ Это может занять несколько минут...")
print("💡 Параметр learning_method='batch' использует все данные за одну итерацию")
print("💡 Для более быстрого обучения можно использовать learning_method='online'")
print()

X_topics = lda.fit_transform(X)
print("\n✅ Обучение завершено!")

# Анализ результатов: выводим 5 самых важных слов для каждой из 10 тем
n_top_words = 5
feature_names = count.get_feature_names_out()

print("\n" + "="*60)
print("ТОП-5 СЛОВ ДЛЯ КАЖДОЙ ТЕМЫ")
print("="*60)

if use_pyprind:
    pbar = pyprind.ProgBar(lda.n_components, title='Анализ тем')

for topic_idx, topic in enumerate(lda.components_):
    print(f'\nТема {(topic_idx + 1)}:')
    top_words = ' '.join([feature_names[i]
                         for i in topic.argsort()[:-n_top_words - 1:-1]])
    print(top_words)
    if use_pyprind:
        pbar.update()

# Интерпретация тем на основе топ-слов
print("\n" + "="*60)
print("ИНТЕРПРЕТАЦИЯ ТЕМ")
print("="*60)
print("""
1. Просто плохие фильмы (не совсем тематическая категория)
2. Фильмы о семье
3. Военные фильмы
4. Художественные фильмы
5. Детективные фильмы
6. Фильмы ужасов
7. Комедии
8. Фильмы, как-то связанные с сериалами
9. Фильмы по книгам
10. Боевики
""")

# Вывод текста отзывов на три фильма из категории "фильмы ужасов" (тема 6, индекс 5)
print("\n" + "="*60)
print("ПРИМЕРЫ ОТЗЫВОВ ИЗ КАТЕГОРИИ 'ФИЛЬМЫ УЖАСОВ' (ТЕМА 6)")
print("="*60)

if use_pyprind:
    pbar = pyprind.ProgBar(3, title='Поиск примеров')

horror = X_topics[:, 5].argsort()[::-1]

for iter_idx, movie_idx in enumerate(horror[:3]):
    print(f'\nФильм ужасов #{(iter_idx + 1)}:')
    print('-' * 60)
    print(df['review'][movie_idx][:300], '...')
    print('-' * 60)
    if use_pyprind:
        pbar.update()

# Дополнительная информация о модели
print("\n" + "="*60)
print("ИНФОРМАЦИЯ О МОДЕЛИ")
print("="*60)
print(f"Форма матрицы components_: {lda.components_.shape}")
print(f"Количество тем: {lda.n_components}")
print(f"Перплексия: {lda.perplexity(X):.2f}")
print(f"Правдоподобие (log-likelihood): {lda.score(X):.2f}")

# Сохранение модели для будущего использования (опционально)
print("\n💾 Сохранение модели...")
if use_pyprind:
    pbar = pyprind.ProgBar(2, title='Сохранение файлов')

import joblib
joblib.dump(lda, 'lda_model.pkl')
if use_pyprind:
    pbar.update()

joblib.dump(count, 'count_vectorizer.pkl')
if use_pyprind:
    pbar.update()

print("✅ Модель и векторизатор сохранены как 'lda_model.pkl' и 'count_vectorizer.pkl'")
