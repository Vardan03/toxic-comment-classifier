import pandas as pd
import numpy as np
from src.models.baseline.tfidf_model import TFIDFModel
from src.config import MODEL_PATH, THRESHOLD

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def load_model():
    model = TFIDFModel()
    model.load(MODEL_PATH)
    return model

def predict(texts: list[str], threshold: float = THRESHOLD) -> pd.DataFrame:
    model = load_model()
    X = model.vectorizer.transform(texts)
    y_proba = model.predict_proba(X)
    y_pred = (y_proba >= threshold).astype(int)
    return pd.DataFrame(y_pred, columns=LABELS)

def predict_proba(texts: list[str]):
    model = load_model()
    X = model.vectorizer.transform(texts)
    return model.predict_proba(X)