import numpy as np
import joblib
import os
from rest_framework.views import APIView
from rest_framework.response import Response
from .serializers import ReviewSerializer

# Singleton para cargar el modelo bajo demanda
class ModelLoader:
    _instance = None
    _model = None
    _nlp = None
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ModelLoader()
        return cls._instance
    
    def load_model(self):
        if self._model is None or self._nlp is None:
            from . import MODEL_PATH
            try:
                _bundle = joblib.load(MODEL_PATH)
                self._model = _bundle["model"]
                self._nlp = _bundle["nlp"]
            except FileNotFoundError as e:
                # En producción, usar un modelo fallback o mostrar un mensaje adecuado
                from rest_framework.exceptions import APIException
                raise APIException(f"Modelo no disponible. Contacte al administrador.")
        return self._model, self._nlp

def _predict(text: str) -> float:
    model, nlp = ModelLoader.get_instance().load_model()
    vec = np.expand_dims(nlp(text).vector, 0)
    return float(model.predict(vec)[0])

class ReviewScoreView(APIView):
    """
    POST  /api/score/
    body: {"text": "..."}
    --> {"score": 3.87}
    """

    serializer_class = ReviewSerializer

    def post(self, request, *args, **kwargs):
        ser = self.serializer_class(data=request.data)
        ser.is_valid(raise_exception=True)
        text = ser.validated_data["text"]
        score = _predict(text)
        return Response({"score": round(score, 2)})
