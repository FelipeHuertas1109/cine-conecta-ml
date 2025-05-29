import os, joblib, pathlib

# Ruta por defecto (dentro de la app)
DEFAULT_PATH = pathlib.Path(__file__).resolve().parent / "models" / "svr_spacy_xz.joblib"
MODEL_PATH   = pathlib.Path(os.getenv("SENTIMENT_MODEL_PATH", DEFAULT_PATH))

bundle = joblib.load(MODEL_PATH)
MODEL, NLP = bundle["model"], bundle["nlp"]
