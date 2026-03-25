import pickle
import pandas as pd
from src.config import MODEL_PATH

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def load_model():
    with open(MODEL_PATH, 'rb') as f:
        model, vectorizer = pickle.load(f)  # model first, vectorizer second
    return model, vectorizer

def predict(texts: list[str]) -> pd.DataFrame:
    model, vectorizer = load_model()

    X = vectorizer.transform(texts)
    predictions = model.predict(X)

    return pd.DataFrame(predictions, columns=LABELS)

def predict_proba(texts: list[str]):
    model, vectorizer = load_model()
    X = vectorizer.transform(texts)
    return model.predict_proba(X)
