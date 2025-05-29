# -*- coding: utf-8 -*-
"""
Predice el puntaje (1-5) de una reseña usando el modelo spaCy+SVR entrenado.

Uso básico:
    python predict_spacy.py --model svr_spacy.joblib --text "Gran película, me encantó"

Modo interactivo:
    python predict_spacy.py --model svr_spacy.joblib
    > escribe reseñas y presiona Enter
    > escribe 'exit' para salir
"""
import argparse
import joblib
import numpy as np

def load(model_path):
    bundle = joblib.load(model_path)
    return bundle["model"], bundle["nlp"]

def predict_one(text: str, model, nlp) -> float:
    vec = np.expand_dims(nlp(text).vector, 0)
    return float(model.predict(vec)[0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="svr_spacy.joblib",
                        help="Ruta al archivo .joblib con el modelo y el pipeline spaCy")
    parser.add_argument("--text", default=None,
                        help="Reseña de película para puntuar (si se omite se abre modo interactivo)")
    args = parser.parse_args()

    model, nlp = load(args.model)

    # Modo línea de comandos
    if args.text is not None:
        score = predict_one(args.text, model, nlp)
        print(f"Puntaje: {score:.2f}")
        return

    # Modo interactivo
    try:
        while True:
            review = input("Escribe una reseña (o 'exit' para salir): ").strip()
            if review.lower() in {"", "exit", "quit"}:
                break
            score = predict_one(review, model, nlp)
            print(f"Puntaje: {score:.2f}")
    except (KeyboardInterrupt, EOFError):
        pass

if __name__ == "__main__":
    main()
