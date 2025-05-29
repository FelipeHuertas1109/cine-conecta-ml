import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from .serializers import ReviewSerializer
from . import MODEL, NLP  # cargados en __init__

def _predict(text: str) -> float:
    vec = np.expand_dims(NLP(text).vector, 0)
    return float(MODEL.predict(vec)[0])

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
