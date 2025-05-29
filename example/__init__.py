import os, joblib

MODEL_PATH = os.getenv("SENTIMENT_MODEL_PATH", "svr_spacy.joblib")

try:
    _bundle = joblib.load(MODEL_PATH)
    MODEL  = _bundle["model"]
    NLP    = _bundle["nlp"]
except FileNotFoundError as e:
    raise RuntimeError(f"No se encontró el modelo en '{MODEL_PATH}'.") from e
