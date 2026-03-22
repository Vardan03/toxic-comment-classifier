import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
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

def evaluate(texts: list[str], y_true) -> None:
    model, vectorizer = load_model()
    X = vectorizer.transform(texts)
    y_pred_proba = model.predict_proba(X)
    y_pred = model.predict(X)

    y_proba = np.column_stack([
        proba[:, 1] if proba.ndim == 2 else proba 
        for proba in y_pred_proba
    ]).T
    print("=" * 40)
    print(f"{'Label':<20} {'ROC AUC':>10} {'F1':>10}")
    print("=" * 40)
    for i, label in enumerate(LABELS):
        auc = roc_auc_score(y_true[:, i], y_proba[:, i])
        f1  = f1_score(y_true[:, i], y_pred[:, i])
        print(f"{label:<20} {auc:>10.4f} {f1:>10.4f}")

    print("=" * 40)
    mean_auc = roc_auc_score(y_true, y_proba, average='macro')
    mean_f1  = f1_score(y_true, y_pred, average='macro')
    print(f"{'MACRO AVG':<20} {mean_auc:>10.4f} {mean_f1:>10.4f}")