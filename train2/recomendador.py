import pandas as pd
import numpy as np

# 1. Cargar el dataset de películas y preprocesar los géneros.
#    Asegúrate de que el archivo 'movies_rows (1).csv' esté en el mismo directorio donde ejecutes este script.
df_movies = pd.read_csv('movies_rows (1).csv')

# El CSV tiene estas columnas principales:
#   - id         → identificador único de la película
#   - title      → título
#   - genre      → género (cadena única, p. ej. 'Acción', 'Drama', etc.)
#   - director, release_date, rating, created_at, updated_at, poster_url (no usados en este ejemplo)

# Renombramos 'id' a 'movie_id' para mantener consistencia con el código anterior
df_movies = df_movies.rename(columns={'id': 'movie_id'})

# Obtener la lista única de géneros que aparecen en todo el dataset
unique_genres = sorted(df_movies['genre'].unique())

# Crear columnas one-hot para cada género (1 si la película pertenece a ese género, 0 en caso contrario)
for genre in unique_genres:
    df_movies[genre] = df_movies['genre'].apply(lambda g: 1 if g == genre else 0)

# Definir el DataFrame de atributos (solo columnas de géneros), indexado por movie_id
df_features = df_movies.set_index('movie_id')[unique_genres]


# 2. Función para construir el perfil de usuario
def construir_perfil_usuario(df_features: pd.DataFrame, user_ratings: dict) -> np.ndarray:
    """
    Dado el DataFrame df_features (películas x géneros) y un diccionario
    user_ratings {movie_id: rating}, devuelve el perfil normalizado del usuario.
    """
    # Filtrar solo las películas que el usuario ha calificado
    rated_ids = [mid for mid in user_ratings if mid in df_features.index]
    if not rated_ids:
        # Si el usuario no ha calificado nada, devolvemos un vector de ceros
        return np.zeros(df_features.shape[1])
    
    df_rated = df_features.loc[rated_ids]
    ratings = np.array([user_ratings[mid] for mid in rated_ids]).reshape(-1, 1)  # (n_rated, 1)
    atributos = df_rated.values  # Matriz de (n_rated, n_features)
    
    # Perfil bruto: suma de rating * vector de atributos
    perfil_bruto = (atributos * ratings).sum(axis=0)  # (n_features,)
    
    # Normalizar para que la magnitud no dependa de la escala de ratings
    norm = np.linalg.norm(perfil_bruto)
    if norm != 0:
        return perfil_bruto / norm
    return perfil_bruto


# 3. Función para recomendar películas
def recomendar_peliculas(df_movies: pd.DataFrame, df_features: pd.DataFrame, user_ratings: dict, top_k: int = 5):
    """
    - Construye el perfil del usuario a partir de user_ratings.
    - Calcula la similitud (dot product) con cada película no vista.
    - Devuelve las top_k películas con mayor similitud.
    
    Parámetros:
    - df_movies: DataFrame original con al menos columnas ['movie_id', 'title', 'genre', ...].
    - df_features: DataFrame one-hot de géneros indexado por movie_id.
    - user_ratings: {movie_id: rating (1–5)} para un usuario específico.
    - top_k: número de recomendaciones a devolver.
    
    Retorna:
    Lista de diccionarios: [{'movie_id': ..., 'title': ..., 'score': ...}, ...]
    """
    # 1. Construir el perfil del usuario
    perfil_usuario = construir_perfil_usuario(df_features, user_ratings)
    
    # 2. Identificar películas no vistas (no calificadas por el usuario)
    vistas = set(user_ratings.keys())
    todos = set(df_features.index)
    no_vistas = sorted(todos - vistas)
    
    # 3. Calcular similitud (dot product) y recolectar (movie_id, score)
    similitudes = []
    for mid in no_vistas:
        vector_atributos = df_features.loc[mid].values
        score = np.dot(perfil_usuario, vector_atributos)
        similitudes.append((mid, score))
    
    # 4. Ordenar por score descendente y tomar los top_k
    top_recomendaciones = sorted(similitudes, key=lambda x: x[1], reverse=True)[:top_k]
    
    # 5. Construir lista final con títulos y puntuación redondeada
    resultados = [
        {
            'movie_id': mid,
            'title': df_movies.loc[df_movies['movie_id'] == mid, 'title'].values[0],
            'score': round(score, 3)
        }
        for mid, score in top_recomendaciones
    ]
    return resultados


# 4. Demostración con un usuario de ejemplo
if __name__ == "__main__":
    # Simulamos los ratings previos del usuario en formato {movie_id: rating}
    # Ajusta estos IDs a los que realmente existan en 'movies_rows (1).csv'
    user_ratings_ejemplo = {
        5: 5,   # Interestelar
        2: 5,   # Inception
        11: 4,   # The Godfather
        1: 3    # Parasite
    }
    
    recomendaciones = recomendar_peliculas(df_movies, df_features, user_ratings_ejemplo, top_k=3)
    
    print("Recomendaciones para el usuario de ejemplo:")
    for rec in recomendaciones:
        print(f"  • {rec['title']} (ID: {rec['movie_id']}), puntuación: {rec['score']}")
