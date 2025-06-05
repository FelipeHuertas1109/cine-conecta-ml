# example/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import APIException
from rest_framework import status

from api.ml import predict                       # ← tu función de puntuación de texto
from .serializers import ReviewSerializer
from .recommendation import (
    get_movies_df,
    get_user_ratings,
    recomendar_peliculas
)


class ReviewScoreView(APIView):
    """
    POST /api/score/
    Body JSON: { "text": "..." }
    Respuesta: { "score": 3.71 }
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


class RecommendMoviesView(APIView):
    """
    GET /api/recommendations/<int:user_id>/
    GET /api/recommendations/<int:user_id>/<int:limit>/
    GET /api/recommendations/<int:user_id>/?limit=<int>
    
    Retorna un JSON con las películas recomendadas para ese usuario,
    basándose en los comentarios (sentiment_score) que ya haya dejado.
    
    Parámetros:
    - user_id: ID del usuario
    - limit (opcional): Cantidad de películas a retornar (por defecto 5)
    """
    def get(self, request, user_id, limit=None):
        # Obtener el límite de películas a retornar
        # Prioridad: parámetro URL > query parameter > valor por defecto (5)
        if limit is None:
            limit = request.GET.get('limit', 5)
        
        try:
            limit = int(limit)
            if limit <= 0:
                return Response(
                    {"error": "El parámetro 'limit' debe ser un número positivo."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            # Limitar a un máximo razonable para evitar sobrecarga
            if limit > 100:
                limit = 100
        except (ValueError, TypeError):
            return Response(
                {"error": "El parámetro 'limit' debe ser un número válido."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 1) Cargar y preprocesar todas las películas
        df_movies, df_features = get_movies_df()

        # 2) Obtener los ratings (sentiment_score) que el usuario ya dejó
        user_ratings = get_user_ratings(user_id)

        # 3) Si no hay comentarios de ese usuario, devolvemos un mensaje y lista vacía
        if not user_ratings:
            return Response(
                {"detail": "El usuario no ha dejado comentarios todavía."},
                status=status.HTTP_200_OK
            )

        # 4) Generar recomendaciones con el límite especificado
        recomendaciones = recomendar_peliculas(df_movies, df_features, user_ratings, top_k=limit)

        # 5) Retornar la lista de recomendaciones con metadatos
        return Response({
            "user_id": user_id,
            "limit": limit,
            "count": len(recomendaciones),
            "recommendations": recomendaciones
        }, status=status.HTTP_200_OK)
