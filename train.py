from src.models.baseline.tfidf_model import TFIDFModel
from src.config import MODEL_PATH, MODELS_PATH
from src.models.deep_learning.rnn_model import RNNModel

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