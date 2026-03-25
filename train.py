from src.models.baseline.tfidf_model import TFIDFModel
from src.config import MODEL_PATH

def train_model(df):
    model = TFIDFModel()
    metrics = model.train(df)
    model.save(MODEL_PATH)

    return metrics