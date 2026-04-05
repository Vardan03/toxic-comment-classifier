import pandas as pd
import numpy as np
import src
from src.models.baseline.tfidf_model import TFIDFModel
from src.config import MODEL_PATH, THRESHOLD, MODELS_PATH, LABEL_COLS
from src.models.deep_learning.rnn_model import RNNModel
from src.models.deep_learning.lstm_model import LSTMModel
from src.models.pretrained.gpt_model import GPT2Model
from src.models.pretrained.bert_model import BERTModel


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


def load_gpt_model():
    model = GPT2Model()
    model.load(MODELS_PATH)
    print("GPT model loaded successfully.")
    return model

def predict_gpt(texts: list[str], threshold: float = THRESHOLD):
    model = load_gpt_model()
    y_pred = model.predict(texts, threshold)
    print("Predictions made successfully.")
    return y_pred

def predict_gpt_proba(texts: list[str]):
    model = load_gpt_model()
    print("Predicting probabilities...")
    return model.predict_proba(texts)


def load_bert_model():
    model = BERTModel()
    model.load(MODELS_PATH)
    print("BERT model loaded successfully.")
    return model


def predict_bert(texts: list[str], threshold: float = THRESHOLD):
    model = load_bert_model()
    y_pred = model.predict(texts, threshold)
    print("Predictions made successfully.")
    return y_pred

def predict_bert_proba(texts: list[str]):
    model = load_bert_model()
    print("Predicting probabilities...")
    return model.predict_proba(texts)