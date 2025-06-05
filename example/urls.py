from django.urls import path
from .views import ReviewScoreView, RecommendMoviesView

urlpatterns = [
    # Ruta existente para puntuar reseñas:
    path("score/", ReviewScoreView.as_view(), name="review-score"),

    # Rutas para recomendaciones:
    # GET  /api/recommendations/<user_id>/
    path(
        "recommendations/<int:user_id>/",
        RecommendMoviesView.as_view(),
        name="recommend-movies"
    ),
    # GET  /api/recommendations/<user_id>/<limit>/
    path(
        "recommendations/<int:user_id>/<int:limit>/",
        RecommendMoviesView.as_view(),
        name="recommend-movies-limit"
    ),
]
