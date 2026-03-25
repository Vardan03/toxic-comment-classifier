import pandas as pd
from src.models.baseline.tfidf_model import TFIDFModel
from src.config import MODEL_PATH

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def load_model():
    model = TFIDFModel()
    model.load(MODEL_PATH)
    return model

def predict(texts: list[str]) -> pd.DataFrame:
    model = load_model()
    X = model.vectorizer.transform(texts)
    predictions = model.predict(X)
    return pd.DataFrame(predictions, columns=LABELS)

def predict_proba(texts: list[str]):
    model = load_model()
    X = model.vectorizer.transform(texts)
    return model.predict_proba(X)