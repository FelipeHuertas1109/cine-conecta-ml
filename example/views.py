# example/views.py

import numpy as np
from rest_framework.views import APIView
from rest_framework.response import Response
from api.ml import MODEL, NLP
from .serializers import ReviewSerializer

class ReviewScoreView(APIView):
    serializer_class = ReviewSerializer

    def post(self, request):
        ser = self.serializer_class(data=request.data)
        ser.is_valid(raise_exception=True)
        text = ser.validated_data["text"]

        vec = np.expand_dims(NLP(text).vector, 0)
        score = float(MODEL.predict(vec)[0])
        return Response({"score": round(score, 2)})
