import numpy as np
import nltk
from fastapi import FastAPI
from sklearn.decomposition import TruncatedSVD

# -------------------- NLTK --------------------
nltk.download('punkt') #  алгоритм для разбиения текста на слова и предложения
nltk.download('averaged_perceptron_tagger_eng') # определение грамматических категорий слов
nltk.download('wordnet') # семантическая сеть, где слова сгруппированы по значению (приведение слов к начальной форме)
nltk.download('omw-1.4') # 
nltk.download('maxent_ne_chunker') # распознавание именованных сущностей
nltk.download('words') # загрузка словаря

# -------------------- APP --------------------
app = FastAPI()

# -------------------- STATE --------------------
documents: list[str] = [] # хранит тексты, загруженные пользователем
tokens: list[list[str]] = [] # список списков строк
vocabulary: list[str] = [] # хранение всех уникальных слов
tfidf_matrix: np.ndarray | None = None 

# -------------------- API --------------------

@app.get("/")
def home():
    return {"status": "ready"}

# ---------- CORPUS LOAD ----------
@app.post("/corpus/load") # эндпоинт для загрузки корпуса
def load_corpus(docs: list[str]): # принимает список строк
    global documents, tokens, vocabulary, tfidf_matrix #  изменяем переменные, объявленные вне функции
# очистка и нормализация документов
    documents = [d.strip().lower() for d in docs if d.strip()]
    tokens = [doc.split() for doc in documents] # токенизация
    vocabulary = sorted(set(w for doc in tokens for w in doc)) # создание словаря уникальных слов
# подсчет статистики
    doc_count = len(tokens)
    word_count = len(vocabulary)

    tf = np.zeros((doc_count, word_count))
    df = np.zeros(word_count)

    for i, doc in enumerate(tokens):
        for word in doc:
            tf[i, vocabulary.index(word)] += 1
        tf[i] /= len(doc)

    for j, word in enumerate(vocabulary):
        df[j] = sum(word in doc for doc in tokens)

    idf = np.log((doc_count + 1) / (df + 1)) + 1
    tfidf_matrix = tf * idf

    return {
        "status": "corpus_loaded",
        "documents": len(documents),
        "vocabulary_size": len(vocabulary)
    }

# ---------- TF-IDF ----------
@app.post("/tf-idf")
def tf_idf():
    return tfidf_matrix.tolist() # возвращение матрицы в обычный список списков

# ---------- BAG OF WORDS ----------
@app.get("/bag-of-words")
def bag_of_words(text: str):
    words = text.lower().split()
    vector = [1 if w in words else 0 for w in vocabulary] # преобразование в вектор
    return vector

# ---------- LSA ----------
@app.post("/lsa")
def lsa(n_components: int = 2):
    n_components = min(max(1, n_components), min(tfidf_matrix.shape))
    svd = TruncatedSVD(n_components=n_components)
    matrix = svd.fit_transform(tfidf_matrix)

    return {
        "matrix": matrix.tolist(),
        "total_variance": float(svd.explained_variance_ratio_.sum())
    }

# -------------------- NLTK --------------------

@app.post("/text_nltk/tokenize")
def tokenize(data: dict):
    text = data["text"]
    tokens = nltk.word_tokenize(text)
    return {"tokens": tokens}

@app.post("/text_nltk/stem")
def stem(data: dict):
    text = data["text"]
    stemmer = nltk.stem.SnowballStemmer("english")
    stems = [stemmer.stem(w) for w in nltk.word_tokenize(text)]
    return {"stems": stems}

@app.post("/text_nltk/lemmatize")
def lemmatize(data: dict):
    text = data["text"]
    lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(w) for w in nltk.word_tokenize(text)]
    return {"lemmas": lemmas}

@app.post("/text_nltk/pos")
def pos(data: dict):
    text = data["text"]
    tags = nltk.pos_tag(nltk.word_tokenize(text))
    return {"pos_tags": tags}

@app.post("/text_nltk/ner")
def ner(data: dict):
    text = data["text"]
    tokens_ = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens_)
    chunks = nltk.ne_chunk(tags)

    entities = []
    for chunk in chunks:
        if hasattr(chunk, "label"):
            words = [w for w, _ in chunk]
            entities.append({
                "entity": " ".join(words),
                "type": chunk.label()
            })
    return {"entities": entities}
