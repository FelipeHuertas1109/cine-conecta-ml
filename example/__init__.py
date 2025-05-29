import os

MODEL_PATH = os.getenv("SENTIMENT_MODEL_PATH", "svr_spacy.joblib")

# Eliminamos la carga del modelo aquí para cargar bajo demanda
# El modelo se cargará solo cuando sea necesario en las vistas
