import numpy as np, joblib, pathlib
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import APIException
from .serializers import ReviewSerializer

# ---------- carga perezosa del joblib ---------- #
MODEL_PATH = pathlib.Path(__file__).resolve().parent / "models" / "svr_spacy_xz.joblib"

class _Model:
    model = nlp = None

def _get_model():
    if _Model.model is None:
        try:
            bundle       = joblib.load(MODEL_PATH)
            _Model.model = bundle["model"]
            _Model.nlp   = bundle["nlp"]
        except FileNotFoundError:
            raise APIException("Modelo no disponible; contacte al administrador.")
    return _Model.model, _Model.nlp
# ------------------------------------------------ #

def _predict(text: str) -> float:
    model, nlp = _get_model()
    vec = np.expand_dims(nlp(text).vector, 0)
    return float(model.predict(vec)[0])

class ReviewScoreView(APIView):
    """
    POST /api/score/   {"text": "..."}  -> {"score": 3.87}
    """

    serializer_class = ReviewSerializer

    def get(self, request):
        return Response({"message": "Usa POST con {'text': '...'} para obtener puntaje"})

    def post(self, request):
        ser = self.serializer_class(data=request.data)
        ser.is_valid(raise_exception=True)
        score = _predict(ser.validated_data["text"])
        return Response({"score": round(score, 2)})
