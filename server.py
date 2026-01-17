import numpy as np
import nltk
from fastapi import FastAPI
from sklearn.decomposition import TruncatedSVD

nltk.download([
    "punkt",
    "averaged_perceptron_tagger",
    "wordnet",
    "omw-1.4",
    "maxent_ne_chunker",
    "words"
])

app = FastAPI(title="NLP Microservice")

# ---------- Загрузка корпуса ----------
def load_corpus():
    try:
        with open("Корпус_Дмитрий.txt", encoding="utf-8") as f:
            docs = []
            for line in f:
                line = line.strip().lower()
                if line:
                    docs.append(line)
            return docs
    except FileNotFoundError:
        return ["файл корпус_дмитрий.txt не найден"]

documents = load_corpus()

# ---------- Предобработка ----------
tokens = []
for doc in documents:
    tokens.append(doc.split())

vocabulary = []
for doc in tokens:
    for word in doc:
        if word not in vocabulary:
            vocabulary.append(word)

doc_count = len(tokens)
word_count = len(vocabulary)

# ---------- TF-IDF (NumPy) ----------
tf = np.zeros((doc_count, word_count))
df = np.zeros(word_count)

for i in range(doc_count):
    for j in range(word_count):
        word = vocabulary[j]
        tf[i][j] = tokens[i].count(word) / len(tokens[i])
        if word in tokens[i]:
            df[j] += 1

idf = np.log((doc_count + 1) / (df + 1)) + 1
tfidf_matrix = tf * idf

# ---------- API ----------
@app.get("/")
def home():
    return {
        "status": "ready",
        "documents_loaded": len(documents)
    }

@app.post("/tf-idf")
def tf_idf():
    return {"matrix": tfidf_matrix.tolist()}

@app.get("/bag-of-words")
def bag_of_words(text):
    words = text.lower().split()
    vector = np.zeros(word_count, dtype=int)

    for i in range(word_count):
        if vocabulary[i] in words:
            vector[i] = 1

    return {"vector": vector.tolist()}

@app.post("/lsa")
def lsa(n_components=2):
    svd = TruncatedSVD(n_components=n_components)
    result = svd.fit_transform(tfidf_matrix)
    return {"matrix": result.tolist()}

# ---------- NLTK ----------
@app.post("/text_nltk/tokenize")
def tokenize(text):
    return nltk.word_tokenize(text)

@app.post("/text_nltk/stem")
def stem(text):
    stemmer = nltk.stem.SnowballStemmer("english")
    result = []
    for word in nltk.word_tokenize(text):
        result.append(stemmer.stem(word))
    return result

@app.post("/text_nltk/lemmatize")
def lemmatize(text):
    lemmatizer = nltk.stem.WordNetLemmatizer()
    result = []
    for word in nltk.word_tokenize(text):
        result.append(lemmatizer.lemmatize(word))
    return result

@app.post("/text_nltk/pos")
def pos(text):
    return nltk.pos_tag(nltk.word_tokenize(text))

@app.post("/text_nltk/ner")
def ner(text):
    tokens = nltk.word_tokenize(text)
    tags = nltk.pos_tag(tokens)
    chunks = nltk.ne_chunk(tags)

    entities = []
    for chunk in chunks:
        if hasattr(chunk, "label"):
            words = []
            for item in chunk:
                words.append(item[0])
            entities.append({
                "entity": " ".join(words),
                "label": chunk.label()
            })

    return entities