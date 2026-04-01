from src.models.baseline.tfidf_model import TFIDFModel
from src.config import MODEL_PATH, MODELS_PATH
from src.models.deep_learning.rnn_model import RNNModel
from src.models.deep_learning.lstm_model import LSTMModel
from src.models.pretrained.gpt_model import GPT2Model

def train_model(df):
    model = TFIDFModel()
    metrics = model.train(df)
    model.save(MODEL_PATH)

    return metrics


def train_rnn_model(X_train, y_train):
    print("Training RNN model...")
    model = RNNModel()
    model = model.fit(X_train, y_train)
    model.save(MODELS_PATH)

    return model

def train_lstm_model(X_train, y_train):
    print("Training LSTM model...")
    model = LSTMModel()
    model = model.fit(X_train, y_train)
    model.save(MODELS_PATH)

    return model


def train_gpt_model(X_train, y_train):
    print("Training GPT model...")
    model = GPT2Model()
    model.fit(X_train, y_train)
    model.save(MODELS_PATH)
    pass