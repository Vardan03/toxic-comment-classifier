import argparse
import numpy as np
from sklearn.model_selection import train_test_split

from src.data.load_data import load_train_data
from src.data.preprocess import preprocess_train
from src.config import (
    TFIDF_PATH, RNN_PATH, LSTM_PATH, BERT_PATH, GPT_PATH,
    RANDOM_STATE, TEST_SIZE, LABEL_COLS
)

from src.models.baseline.tfidf_model import TFIDFModel
from src.models.deep_learning.rnn_model import RNNModel
from src.models.deep_learning.lstm_model import LSTMModel
from src.models.pretrained.bert_model import BERTModel
from src.models.pretrained.gpt_model import GPT2Model


def train_tfidf(df, **kwargs):
    print("Training TF-IDF model...")
    model = TFIDFModel()
    metrics = model.train(df)
    model.save(TFIDF_PATH)
    return metrics


def train_rnn(X_train, y_train, X_val, y_val, glove_path=None, **kwargs):
    print("Training RNN model...")
    model = RNNModel(
        vocab_size  = 30_000,
        max_seq_len = 200,
        embed_dim   = 100,
        hidden_dim  = 256,
        num_layers  = 2,
        dropout     = 0.3,
        lr          = 1e-3,
        batch_size  = 128,
        epochs      = 20,
        device      = "auto"
    )
    model.fit(X_train, y_train, X_val, y_val, glove_path=glove_path)
    model.save(RNN_PATH)
    return model


def train_lstm(X_train, y_train, X_val, y_val, glove_path=None, **kwargs):
    print("Training LSTM model...")
    model = LSTMModel(
        vocab_size  = 30_000,
        max_seq_len = 200,
        embed_dim   = 100,
        hidden_dim  = 256,
        num_layers  = 3,
        dropout     = 0.5,
        lr          = 1e-3,
        batch_size  = 128,
        epochs      = 20,
        device      = "auto"
    )
    model.fit(X_train, y_train, X_val, y_val, glove_path=glove_path)
    model.save(LSTM_PATH)
    return model


def train_bert(X_train, y_train, X_val, y_val, **kwargs):
    print("Training BERT model...")
    model = BERTModel()
    model.fit(X_train, y_train, X_val, y_val)
    model.save(BERT_PATH)
    return model


def train_gpt(X_train, y_train, X_val, y_val, **kwargs):
    print("Training GPT model...")
    model = GPT2Model()
    model.fit(X_train, y_train, X_val, y_val)
    model.save(GPT_PATH)
    return model


TRAINERS = {
    "tfidf": train_tfidf,
    "rnn":   train_rnn,
    "lstm":  train_lstm,
    "bert":  train_bert,
    "gpt":   train_gpt,
}


def main():
    parser = argparse.ArgumentParser(description="Train a toxic comment classifier model")
    parser.add_argument(
        "--model",
        choices=TRAINERS.keys(),
        required=True,
        help="Model to train: tfidf, rnn, lstm, bert, gpt"
    )
    parser.add_argument(
        "--glove",
        type=str,
        default=None,
        help="Path to GloVe embeddings file (for rnn/lstm only)"
    )
    args = parser.parse_args()

    # load and preprocess data
    print("Loading data...")
    df = load_train_data()
    df = preprocess_train(df)

    if args.model == "tfidf":
        # tfidf handles its own split internally
        TRAINERS["tfidf"](df)
    else:
        # prepare data for deep learning models
        X = [str(x) for x in df['comment_text'].tolist()]  # fix NaN
        y = df[LABEL_COLS].values

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        print(f"Train: {len(X_train)} | Val: {len(X_val)}")

        TRAINERS[args.model](
            X_train    = X_train,
            y_train    = y_train,
            X_val      = X_val,
            y_val      = y_val,
            glove_path = args.glove,
        )

    print(f"✅ {args.model} training complete!")


if __name__ == "__main__":
    main()