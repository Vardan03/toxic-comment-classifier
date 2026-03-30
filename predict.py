import pandas as pd
import numpy as np
from src.models.baseline.tfidf_model import TFIDFModel
from src.config import MODEL_PATH, THRESHOLD, MODELS_PATH, LABEL_COLS
from src.models.deep_learning.rnn_model import RNNModel
from src.models.deep_learning.lstm_model import LSTMModel

def load_model():
    model = TFIDFModel()
    model.load(MODEL_PATH)
    return model

def predict(texts: list[str], threshold: float = THRESHOLD) -> pd.DataFrame:
    model = load_model()
    X = model.vectorizer.transform(texts)
    y_proba = model.predict_proba(X)
    y_pred = (y_proba >= threshold).astype(int)
    return pd.DataFrame(y_pred, columns=LABEL_COLS)

def predict_proba(texts: list[str]):
    model = load_model()
    X = model.vectorizer.transform(texts)
    return model.predict_proba(X)

def load_rnn_model():
    model = RNNModel()
    model.load(MODELS_PATH)
    return model


def predict_rnn(texts: list[str], threshold: float = THRESHOLD) -> pd.DataFrame:
    model = load_rnn_model()
    y_pred = model.predict(texts, threshold)
    return y_pred

def predict_rnn_proba(texts: list[str]):
    model = load_rnn_model()
    return model.predict_proba(texts)

def load_lstm_model():
    model = LSTMModel()
    model.load(MODELS_PATH)
    return model

def predict_lstm(texts: list[str], threshold: float = THRESHOLD) -> pd.DataFrame:
    model = load_lstm_model()
    y_pred = model.predict(texts, threshold)
    return y_pred

def predict_lstm_proba(texts: list[str]):
    model = load_lstm_model()
    return model.predict_proba(texts)