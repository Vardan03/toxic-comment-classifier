import os
import pickle
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.config import LABEL_COLS, MODEL_PATH, MODELS_PATH, THRESHOLD
from src.models.baseline.tfidf_model import TFIDFModel
from src.models.deep_learning.bert_model import BERTClassifier
from src.models.deep_learning.rnn_model import RNNModel


def _load_saved_config() -> Dict[str, Any]:
    config_path = os.path.join(MODELS_PATH, "config.pkl")
    if not os.path.exists(config_path):
        return {}
    with open(config_path, "rb") as f:
        config = pickle.load(f)
    if not isinstance(config, dict):
        return {}
    return config


def _resolve_model_type(config: Optional[Dict[str, Any]] = None) -> str:
    merged = _load_saved_config()
    if config:
        merged.update(config)

    model_type = merged.get("model")
    if model_type:
        return str(model_type).lower()

    if os.path.exists(os.path.join(MODELS_PATH, "weights_rnn.pt")):
        return "rnn"
    if os.path.exists(os.path.join(MODELS_PATH, "weights_bert.pt")):
        return "bert"
    return "rnn"


def _build_deep_learning_model(model_type: str):
    if model_type == "rnn":
        return RNNModel()
    if model_type == "bert":
        return BERTClassifier()
    if model_type == "lstm":
        raise NotImplementedError(
            "LSTM selection was requested, but src/models/deep_learning/lstm_model.py "
            "is empty in the current workspace."
        )
    raise ValueError(f"Unsupported deep learning model: {model_type}")


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


def predict_rnn(texts: list[str], threshold: float = THRESHOLD) -> np.ndarray:
    model = load_rnn_model()
    return model.predict(texts, threshold)


def predict_rnn_proba(texts: list[str]):
    model = load_rnn_model()
    return model.predict_proba(texts)


def load_bert_model():
    model = BERTClassifier()
    model.load(MODELS_PATH)
    return model


def predict_bert(texts: list[str], threshold: float = THRESHOLD) -> np.ndarray:
    model = load_bert_model()
    return model.predict(texts, threshold)


def predict_bert_proba(texts: list[str]):
    model = load_bert_model()
    return model.predict_proba(texts)


def load_deep_learning_model(config: Optional[Dict[str, Any]] = None):
    model_type = _resolve_model_type(config)
    model = _build_deep_learning_model(model_type)
    model.load(MODELS_PATH)
    return model


def predict_deep_learning(
    texts: list[str],
    threshold: float = THRESHOLD,
    config: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    model = load_deep_learning_model(config)
    return model.predict(texts, threshold)


def predict_deep_learning_proba(
    texts: list[str],
    config: Optional[Dict[str, Any]] = None,
):
    model = load_deep_learning_model(config)
    return model.predict_proba(texts)
