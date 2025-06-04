from django.urls import path
from .views import ReviewScoreView

urlpatterns = [
    path("score/", ReviewScoreView.as_view()),
]
