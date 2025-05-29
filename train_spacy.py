# -*- coding: utf-8 -*-
"""
Entrena un regresor de sentimientos 1-5 usando embeddings de spaCy.
Usage:
    python train_spacy.py --data reviews_movies_1000.csv --model svr_spacy.joblib
"""
import argparse, joblib, numpy as np, pandas as pd, spacy
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

def vectorize_texts(texts, nlp):
    """Convierte una lista de textos en una matriz (n_docs x 300) con doc.vector"""
    return np.vstack([nlp(t).vector for t in texts])

def main(data_path: str, out_model: str, algo: str = "svr"):
    # 1. Carga el dataset
    df = pd.read_csv(data_path)
    train_df = df[df["split"] == "train"]
    valid_df = df[df["split"] == "valid"]
    test_df  = df[df["split"] == "test"]

    # 2. Carga spaCy
    nlp = spacy.load("es_core_news_md")   # usa _sm o _lg si prefieres

    # 3. Vectoriza
    X_train = vectorize_texts(train_df["text"], nlp)
    y_train = train_df["score"].values
    X_valid = vectorize_texts(valid_df["text"], nlp)
    y_valid = valid_df["score"].values
    X_test  = vectorize_texts(test_df["text"],  nlp)
    y_test  = test_df["score"].values

    # 4. Modelo
    if algo == "svr":
        model = SVR(kernel="rbf", C=2.0, epsilon=0.1)
    else:
        model = Ridge(alpha=1.0)

    model.fit(np.vstack([X_train, X_valid]), np.hstack([y_train, y_valid]))

    # 5. Evaluación
    preds = model.predict(X_test)
    mae  = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f"Test MAE : {mae:.3f}")
    print(f"Test RMSE: {rmse:.3f}")

    # 6. Guarda el modelo y el pipeline spaCy
    joblib.dump({"model": model, "nlp": nlp}, out_model)
    print(f"Modelo guardado en {out_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default="reviews_movies_1000.csv")
    parser.add_argument("--model", default="svr_spacy.joblib")
    parser.add_argument("--algo",  choices=["svr", "ridge"], default="svr",
                        help="Algoritmo de regresión a usar")
    args = parser.parse_args()
    main(args.data, args.model, args.algo)
