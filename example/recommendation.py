# example/recommendation.py

import pandas as pd
import numpy as np
from .models import Movie, Comment

def get_movies_df():
    """
    1) Obtiene todas las películas desde la tabla Movie.
    2) Construye un DataFrame pandas con columnas one-hot para cada género.
    Retorna:
        - df_movies: DataFrame original con columnas ['movie_id','title','genre', ... géneros one-hot ...]
        - df_features: DataFrame indexado por movie_id que contiene solo las columnas one-hot de géneros.
    """
    # 1.1. Queryset de todas las películas: solo traemos id, title y genre
    qs = Movie.objects.all().values('id', 'title', 'genre')
    
    # 1.2. Convertimos a DataFrame
    df = pd.DataFrame.from_records(qs)
    
    # 1.3. Renombramos la columna 'id' → 'movie_id' para mayor claridad
    df = df.rename(columns={'id': 'movie_id'})
    
    # 1.4. Obtenemos la lista única de géneros
    unique_genres = sorted(df['genre'].dropna().unique())
    
    # 1.5. Creamos una columna one-hot por cada género
    for genre in unique_genres:
        df[genre] = df['genre'].apply(lambda g: 1 if g == genre else 0)
    
    # 1.6. Construimos df_features: indexado por movie_id, solo con las columnas de géneros one-hot
    df_features = df.set_index('movie_id')[unique_genres]
    
    return df, df_features


def get_user_ratings(user_id: int) -> dict:
    """
    Consulta todos los comentarios de ese usuario y devuelve un diccionario
      { movie_id: rating }
    usando Comment.sentiment_score como valor numérico de rating.
    """
    # Filtramos todos los comentarios que el usuario ha dejado
    qs = Comment.objects.filter(user_id=user_id).values_list('movie_id', 'sentiment_score')
    
    # Convertimos la lista de tuplas [(movie_id, sentiment_score), ...] a dict
    return {movie_id: float(score) for (movie_id, score) in qs}


def construir_perfil_usuario(df_features: pd.DataFrame, user_ratings: dict) -> np.ndarray:
    """
    Dado el DataFrame df_features (películas x géneros) y un diccionario
      user_ratings { movie_id: rating }, construye el perfil normalizado del usuario.
    Retorna:
      - perfil_normalizado: numpy array de tamaño (n_features,)
    """
    # 1. Filtrar solo las películas que el usuario ha calificado y existen en df_features
    rated_ids = [mid for mid in user_ratings.keys() if mid in df_features.index]
    if not rated_ids:
        # Si el usuario no ha puntuado nada, devolvemos un vector de ceros
        return np.zeros(df_features.shape[1])
    
    # 2. Extraer la matriz de atributos (géneros) para esas películas calificadas
    df_rated = df_features.loc[rated_ids]                      # shape = (n_rated, n_features)
    ratings = np.array([user_ratings[mid] for mid in rated_ids]).reshape(-1, 1)  # (n_rated, 1)
    atributos = df_rated.values                                 # (n_rated, n_features)
    
    # 3. Perfil bruto = suma de rating * vector_atributos para cada película
    perfil_bruto = (atributos * ratings).sum(axis=0)            # (n_features,)
    
    # 4. Normalizamos para que no dependa de la escala absoluta de los ratings
    norm = np.linalg.norm(perfil_bruto)
    if norm != 0:
        perfil_normalizado = perfil_bruto / norm
    else:
        perfil_normalizado = perfil_bruto
    
    return perfil_normalizado  # numpy.ndarray (n_features,)


def recomendar_peliculas(
    df_movies: pd.DataFrame,
    df_features: pd.DataFrame,
    user_ratings: dict,
    top_k: int = 5
) -> list:
    """
    — Construye el perfil del usuario usando 'construir_perfil_usuario'.
    — Calcula la similitud (dot product) entre el perfil y cada película no vista.
    — Devuelve las 'top_k' películas con mayor similitud.

    Parámetros:
      - df_movies: DataFrame completo (incluye columnas movie_id, title, genre, ...)
      - df_features: DataFrame indexado por movie_id con únicamente columnas one-hot de géneros.
      - user_ratings: diccionario { movie_id: rating } para el usuario.
      - top_k: número de recomendaciones a retornar.

    Retorna:
      Una lista de diccionarios en formato:
      [
        { 'movie_id': <int>, 'title': <str>, 'score': <float> },
        ...
      ]
    """
    # 1) Construir perfil del usuario
    perfil_usuario = construir_perfil_usuario(df_features, user_ratings)
    
    # 2) Identificar películas no vistas (no calificadas por el usuario)
    vistas = set(user_ratings.keys())
    todos = set(df_features.index)
    no_vistas = sorted(todos - vistas)
    
    # 3) Para cada película no vista, calcular similitud (producto punto)
    similitudes = []
    for mid in no_vistas:
        vector_atributos = df_features.loc[mid].values   # vector de géneros
        score = float(np.dot(perfil_usuario, vector_atributos))
        similitudes.append((mid, score))
    
    # 4) Ordenar por score descendente y tomar los top_k
    top_recomendaciones = sorted(similitudes, key=lambda x: x[1], reverse=True)[:top_k]
    
    # 5) Construir lista final con títulos y puntuaciones redondeadas
    resultados = []
    for (mid, score) in top_recomendaciones:
        title = df_movies.loc[df_movies['movie_id'] == mid, 'title'].values[0]
        resultados.append({
            'movie_id': mid,
            'title': title,
            'score': round(score, 3)
        })
    
    return resultados
