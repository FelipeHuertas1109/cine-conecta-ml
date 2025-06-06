import pandas as pd
import joblib
import nltk
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np

# Descargar recursos de NLTK si no están disponibles
try:
    nltk.data.find("tokenizers/punkt_tab/spanish")
except LookupError:
    nltk.download("punkt_tab")
try:
    stopwords.words("spanish")
except LookupError:
    nltk.download("stopwords")

_stop = set(stopwords.words("spanish"))

def tokenizer_nltk(texto):
    return [t for t in word_tokenize(texto.lower(), language="spanish")
            if t.isalpha() and t not in _stop]

# 1) Cargar datos
df = pd.read_csv("reviews_movies_teen_1000.csv", encoding="utf-8")
X, y = df["text"].astype(str), df["score"].astype(float)

# 2) División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% para test
    random_state=42,    # Para reproducibilidad
    stratify=None       # Para regresión no usamos stratify
)

print(f"Tamaño del conjunto de entrenamiento: {len(X_train)}")
print(f"Tamaño del conjunto de prueba: {len(X_test)}")

# 3) Pipeline
pipe = make_pipeline(
    TfidfVectorizer(tokenizer=tokenizer_nltk,
                    ngram_range=(1,2),
                    max_features=20_000,
                    strip_accents="unicode"),
    StandardScaler(with_mean=False),
    SVR(C=2.0)
)

# 4) Entrenar el modelo
print("\nEntrenando modelo...")
pipe.fit(X_train, y_train)

# 5) Hacer predicciones
print("Realizando predicciones...")
y_pred_train = pipe.predict(X_train)
y_pred_test = pipe.predict(X_test)

# 6) Evaluar el modelo
def evaluar_modelo(y_true, y_pred, conjunto=""):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n=== Métricas {conjunto} ===")
    print(f"MSE (Error Cuadrático Medio): {mse:.4f}")
    print(f"RMSE (Raíz del Error Cuadrático Medio): {rmse:.4f}")
    print(f"MAE (Error Absoluto Medio): {mae:.4f}")
    print(f"R² (Coeficiente de Determinación): {r2:.4f}")
    
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2}

# Evaluar en conjunto de entrenamiento
metricas_train = evaluar_modelo(y_train, y_pred_train, "ENTRENAMIENTO")

# Evaluar en conjunto de prueba
metricas_test = evaluar_modelo(y_test, y_pred_test, "PRUEBA")

# 7) Análisis de overfitting
print(f"\n=== Análisis de Overfitting ===")
print(f"Diferencia R² (Train - Test): {metricas_train['R2'] - metricas_test['R2']:.4f}")
print(f"Diferencia RMSE (Test - Train): {metricas_test['RMSE'] - metricas_train['RMSE']:.4f}")

if metricas_train['R2'] - metricas_test['R2'] > 0.1:
    print("⚠️  Posible overfitting detectado")
else:
    print("✅ No hay signos claros de overfitting")

# 8) Ejemplos de predicciones
print(f"\n=== Ejemplos de Predicciones ===")
for i in range(min(5, len(X_test))):
    idx = X_test.iloc[i:i+1].index[0]
    texto = X_test.iloc[i][:100] + "..." if len(X_test.iloc[i]) > 100 else X_test.iloc[i]
    real = y_test.iloc[i]
    pred = y_pred_test[i]
    print(f"\nTexto: {texto}")
    print(f"Score real: {real:.2f} | Score predicho: {pred:.2f} | Error: {abs(real-pred):.2f}")

# 9) Guardar modelo y métricas
modelo_datos = {
    "model": pipe,
    "metricas_train": metricas_train,
    "metricas_test": metricas_test,
    "train_size": len(X_train),
    "test_size": len(X_test)
}

joblib.dump(modelo_datos, "svr_nltk_tfidf_evaluated.joblib")
print(f"\n✅ Modelo y métricas guardados en: svr_nltk_tfidf_evaluated.joblib")

# 10) Resumen final
print(f"\n=== RESUMEN FINAL ===")
print(f"Mejor métrica R² en test: {metricas_test['R2']:.4f}")
print(f"Error promedio (MAE) en test: {metricas_test['MAE']:.4f}")
print(f"El modelo explica el {metricas_test['R2']*100:.1f}% de la varianza en los datos de prueba")