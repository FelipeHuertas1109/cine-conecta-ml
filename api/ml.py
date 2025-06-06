# api/ml.py
"""
Carga del modelo NLTK-TFIDF + SVR y función predict(text) → score

•  Sin variables de entorno
•  Corrige el error clásico de Joblib
      AttributeError: can't get attribute 'tokenizer_nltk' on <module '__main__' …>
  registrando la función dentro de sys.modules['__main__'] ANTES de des-pickle-ar.
"""

from __future__ import annotations
import sys, types, pathlib, joblib
from functools import lru_cache
from django.core.exceptions import ImproperlyConfigured
from sklearn.pipeline import Pipeline               # sólo para anotación

# ------------------------------------------------------------------
# 1) —— Ruta del modelo (ajústala si lo mueves de carpeta)
# ------------------------------------------------------------------
MODEL_PATH = (
    pathlib.Path(__file__).resolve().parent  # api/
    / "models" / "svr_nltk_tfidf_evaluated.joblib"     # api/models/...
)

# ------------------------------------------------------------------
# 2) —— Descargar (o comprobar) recursos NLTK necesarios
# ------------------------------------------------------------------
import nltk

for res in ("stopwords", "punkt", "punkt_tab"):
    # Si el recurso ya está presente, nltk.download() no hace nada
    nltk.download(res, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

_STOP_ES = set(stopwords.words("spanish"))

def tokenizer_nltk(texto: str) -> list[str]:
    """Tokenizador usado al entrenar (minúsculas, solo alpha, sin stop-words)."""
    tokens = word_tokenize(texto.lower(), language="spanish")
    return [t for t in tokens if t.isalpha() and t not in _STOP_ES]

# ------------------------------------------------------------------
# 3) —— Hack: publicar la función en «__main__» para que Joblib la encuentre
# ------------------------------------------------------------------
_main_stub = types.ModuleType("__main__")
_main_stub.tokenizer_nltk = tokenizer_nltk
sys.modules["__main__"] = _main_stub

# ------------------------------------------------------------------
# 4) —— Cargar el Pipeline
# ------------------------------------------------------------------
try:
    _bundle: dict[str, Pipeline] = joblib.load(MODEL_PATH)   # {'model': …}
except FileNotFoundError as exc:
    raise ImproperlyConfigured(
        f"No se encontró el modelo en {MODEL_PATH}. "
        f"Asegúrate de moverlo allí o cambia MODEL_PATH."
    ) from exc
except Exception as exc:
    raise ImproperlyConfigured(
        f"No se pudo cargar el modelo «{MODEL_PATH}»: {exc}"
    ) from exc

_PIPE: Pipeline = _bundle["model"]     # alias interno

# ------------------------------------------------------------------
# 5) —— API pública
# ------------------------------------------------------------------
@lru_cache(maxsize=8)
def _predict_cached(text: str) -> float:
    """Predicción con memoización sencilla (últimas 8 llamadas)."""
    return float(_PIPE.predict([text])[0])

def predict(text: str) -> float:
    """
    Calcula la puntuación de la reseña.

    Parameters
    ----------
    text : str
        Reseña en castellano.

    Returns
    -------
    float
        Puntaje 1-5 (según tu dataset).
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("El texto debe ser una cadena no vacía")
    return _predict_cached(text.strip())
