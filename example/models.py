# example/models.py

from django.db import models


class User(models.Model):
    """
    Modelo que “apunta” a la tabla existente 'users'.
    Campos según tu esquema: id INT8 PK, name, email, password, role.
    """
    id = models.BigIntegerField(primary_key=True)
    name = models.CharField(max_length=255)
    email = models.EmailField(max_length=255, unique=True)
    password = models.CharField(max_length=255)
    role = models.CharField(max_length=50)

    class Meta:
        managed = False
        db_table = 'users'
        ordering = ['id']

    def __str__(self):
        return f"{self.name} ({self.email})"


class Movie(models.Model):
    """
    Modelo que mapea la tabla existente 'movies'.
    Campos según tu esquema: 
      id INT8 PK, title TEXT, description TEXT, genre TEXT, director TEXT, 
      release_date TIMESTAMPZ, rating NUMERIC, created_at TIMESTAMPZ, updated_at TIMESTAMPZ, poster_url TEXT.
    """
    id = models.BigIntegerField(primary_key=True)
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True, null=True)
    genre = models.CharField(max_length=100, blank=True, null=True)
    director = models.CharField(max_length=100, blank=True, null=True)
    release_date = models.DateTimeField(blank=True, null=True)
    rating = models.FloatField(blank=True, null=True)
    created_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(blank=True, null=True)
    poster_url = models.TextField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'movies'
        ordering = ['title']

    def __str__(self):
        return self.title


class Comment(models.Model):
    """
    Modelo que mapea la tabla existente 'comments'.
    Campos según tu esquema:
      id INT8 PK, user_id INT8 FK(users.id), movie_id INT8 FK(movies.id),
      content TEXT, created_at TIMESTAMPZ, updated_at TIMESTAMPZ, 
      sentiment TEXT, sentiment_score NUMERIC.
    """
    id = models.BigIntegerField(primary_key=True)
    
    # ForeignKey a User: db_column='user_id' le dice que la columna en BD es user_id
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        db_column='user_id',
        related_name='comments'
    )
    
    # ForeignKey a Movie: db_column='movie_id' le dice que la columna en BD es movie_id
    movie = models.ForeignKey(
        Movie,
        on_delete=models.CASCADE,
        db_column='movie_id',
        related_name='comments'
    )
    
    content = models.TextField(blank=True, null=True)
    created_at = models.DateTimeField(blank=True, null=True)
    updated_at = models.DateTimeField(blank=True, null=True)
    
    # Por si quieres usar el texto categorizado (p.ej. "positivo", "negativo")
    sentiment = models.CharField(max_length=50, blank=True, null=True)
    
    # Este campo numérico (NUMERIC en BD) lo usamos como “rating” en la recomendación
    sentiment_score = models.FloatField(blank=True, null=True)

    class Meta:
        managed = False
        db_table = 'comments'
        ordering = ['-created_at']

    def __str__(self):
        return f"User {self.user_id} → Movie {self.movie_id} (Score: {self.sentiment_score})"
