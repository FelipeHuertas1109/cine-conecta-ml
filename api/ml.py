# api/ml.py

import os
import joblib
import pathlib
import spacy
from spacy import util
from spacy.cli import download as spacy_download
from django.conf import settings

# 1) spaCy: carga el modelo pequeño y excluye componentes para aligerar memoria
MODEL_NAME = "es_core_news_sm"
if not util.is_package(MODEL_NAME):
    spacy_download(MODEL_NAME)

NLP = spacy.load(
    MODEL_NAME,
    exclude=[
        "parser",
        "ner",
        "tagger",
        "attribute_ruler",
        "lemmatizer",
    ],
)

# 2) SVR: construye la ruta al .joblib que realmente tienes en disco
BASE = pathlib.Path(settings.BASE_DIR)
DEFAULT_PATH = BASE / "example" / "models" / "svr_spacy_sm.joblib"
MODEL_PATH = pathlib.Path(os.getenv("SENTIMENT_MODEL_PATH", DEFAULT_PATH))

# Aquí sí te devuelve directamente el SVR, no un dict
MODEL = joblib.load(MODEL_PATH)
