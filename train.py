from typing import Any, Dict, Optional

from src.config import MODEL_PATH, MODELS_PATH
from src.models.baseline.tfidf_model import TFIDFModel
from src.models.deep_learning.bert_model import BERTClassifier
from src.models.deep_learning.rnn_model import RNNModel


def _normalize_model_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    normalized = dict(config or {})
    normalized.setdefault("model", "rnn")
    return normalized


def _build_deep_learning_model(config: Optional[Dict[str, Any]] = None):
    config = _normalize_model_config(config)
    model_type = config["model"].lower()

    common_kwargs = {
        key: value
        for key, value in config.items()
        if key != "model" and value is not None
    }

    if model_type == "rnn":
        return RNNModel(**common_kwargs)
    if model_type == "bert":
        return BERTClassifier(**common_kwargs)
    if model_type == "lstm":
        raise NotImplementedError(
            "LSTM selection was requested, but src/models/deep_learning/lstm_model.py "
            "is empty in the current workspace."
        )
    raise ValueError(f"Unsupported deep learning model: {model_type}")


def train_model(df):
    model = TFIDFModel()
    metrics = model.train(df)
    model.save(MODEL_PATH)
    return metrics


def train_deep_learning_model(
    X_train,
    y_train,
    config: Optional[Dict[str, Any]] = None,
    X_val=None,
    y_val=None,
):
    resolved_config = _normalize_model_config(config)
    model = _build_deep_learning_model(resolved_config)
    print(f"Training {resolved_config['model'].upper()} model...")
    model = model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    model.save(MODELS_PATH)
    return model


def train_rnn_model(X_train, y_train, X_val=None, y_val=None):
    return train_deep_learning_model(
        X_train,
        y_train,
        config={"model": "rnn"},
        X_val=X_val,
        y_val=y_val,
    )


def train_bert_model(X_train, y_train, X_val=None, y_val=None, config=None):
    merged_config = {"model": "bert"}
    if config:
        merged_config.update(config)
    return train_deep_learning_model(
        X_train,
        y_train,
        config=merged_config,
        X_val=X_val,
        y_val=y_val,
    )
