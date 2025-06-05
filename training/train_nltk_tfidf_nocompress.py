# train_nltk_tfidf_nocompress.py
import pandas as pd, joblib, nltk
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

try:
    nltk.data.find("tokenizers/punkt_tab/spanish")
except LookupError:
    nltk.download("punkt_tab")
try:
    stopwords.words("spanish")
except LookupError:
    nltk.download("stopwords")

_stop = set(stopwords.words("spanish"))
def tokenizer_nltk(texto):
    return [t for t in word_tokenize(texto.lower(), language="spanish")
            if t.isalpha() and t not in _stop]

# 1) Datos
df = pd.read_csv("reviews_movies_teen_1000.csv", encoding="utf-8")
X, y = df["text"].astype(str), df["score"].astype(float)

# 2) Pipeline
pipe = make_pipeline(
    TfidfVectorizer(tokenizer=tokenizer_nltk,
                    ngram_range=(1,2),
                    max_features=20_000,
                    strip_accents="unicode"),
    StandardScaler(with_mean=False),
    SVR(C=2.0)
)

print("Entrenando…")
pipe.fit(X, y)

# 3) Guardar SIN compresión
joblib.dump({"model": pipe}, "svr_nltk_tfidf.joblib")
print("Archivo generado: svr_nltk_tfidf.joblib")
