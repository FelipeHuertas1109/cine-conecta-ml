from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import APIException

from api.ml import predict                       # ← nuestra función pred
from .serializers import ReviewSerializer


class ReviewScoreView(APIView):
    """
    POST /api/score/
    Body JSON: {"text": "..." }
    Respuesta : {"score": 3.71}
    """

    serializer_class = ReviewSerializer

    def post(self, request):
        ser = self.serializer_class(data=request.data)
        ser.is_valid(raise_exception=True)

        try:
            score = predict(ser.validated_data["text"])
        except Exception as exc:
            raise APIException(f"No se pudo puntuar la reseña: {exc}")

        return Response({"score": round(score, 2)})
