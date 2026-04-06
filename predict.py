import argparse
import pandas as pd
import numpy as np
from src.config import (
    TFIDF_PATH, RNN_PATH, LSTM_PATH, BERT_PATH, GPT_PATH,
    THRESHOLD, LABEL_COLS
)
from src.models.baseline.tfidf_model import TFIDFModel
from src.models.deep_learning.rnn_model import RNNModel
from src.models.deep_learning.lstm_model import LSTMModel
from src.models.pretrained.bert_model import BERTModel
from src.models.pretrained.gpt_model import GPT2Model


# ─────────────────────────────────────────────
# LOADERS
# ─────────────────────────────────────────────

def load_tfidf():
    model = TFIDFModel()
    model.load(TFIDF_PATH)
    return model

def load_rnn():
    model = RNNModel()
    model.load(RNN_PATH)
    return model

def load_lstm():
    model = LSTMModel()
    model.load(LSTM_PATH)
    return model

def load_bert():
    model = BERTModel()
    model.load(BERT_PATH)
    return model

def load_gpt():
    model = GPT2Model()
    model.load(GPT_PATH)
    return model


LOADERS = {
    "tfidf": load_tfidf,
    "rnn":   load_rnn,
    "lstm":  load_lstm,
    "bert":  load_bert,
    "gpt":   load_gpt,
}


# ─────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────

def predict(texts: list[str], model_name: str, threshold: float = THRESHOLD) -> pd.DataFrame:
    """
    Return binary predictions for each label.

    Args:
        texts      : list of comment texts
        model_name : one of tfidf, rnn, lstm, bert, gpt
        threshold  : decision threshold, default from config
    Returns:
        pd.DataFrame of shape (n, 6) with binary predictions
    """
    model = LOADERS[model_name]()

    if model_name == "tfidf":
        X = model.vectorizer.transform(texts)
        y_proba = model.predict_proba(X)
        y_pred  = (y_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(texts, threshold=threshold)

    return pd.DataFrame(y_pred, columns=LABEL_COLS)


def predict_proba(texts: list[str], model_name: str) -> np.ndarray:
    """
    Return predicted probabilities for each label.

    Args:
        texts      : list of comment texts
        model_name : one of tfidf, rnn, lstm, bert, gpt
    Returns:
        np.ndarray of shape (n, 6)
    """
    model = LOADERS[model_name]()

    if model_name == "tfidf":
        X = model.vectorizer.transform(texts)
        return model.predict_proba(X)
    else:
        return model.predict_proba(texts)


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run predictions with a trained model")
    parser.add_argument(
        "--model",
        choices=LOADERS.keys(),
        required=True,
        help="Model to use: tfidf, rnn, lstm, bert, gpt"
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Text to classify"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help=f"Decision threshold, default {THRESHOLD}"
    )
    args = parser.parse_args()

    texts = [args.text]
    preds = predict(texts, args.model, args.threshold)
    proba = predict_proba(texts, args.model)

    print(f"\n📝 Text: {args.text}")
    print(f"\n{'Label':<20} {'Probability':>12} {'Prediction':>12}")
    print("=" * 46)
    for i, label in enumerate(LABEL_COLS):
        prob = proba[0][i]
        pred = preds[label].iloc[0]
        flag = "🚨 TOXIC" if pred == 1 else "✅ CLEAN"
        print(f"{label:<20} {prob:>12.4f} {flag:>12}")


if __name__ == "__main__":
    main()