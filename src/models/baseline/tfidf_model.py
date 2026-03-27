import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from src.utils.metrics import compute_metrics
from src.models.base_model import BaseModel
from src.features.tfidf import fit_tfidf, transform_tfidf
from src.config import MODEL_PATH, RANDOM_STATE, TEST_SIZE

LABEL_COLUMNS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


class TFIDFModel(BaseModel):
    def __init__(self):
        self.model = OneVsRestClassifier(
            LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight="balanced")
        )
        self.vectorizer = None

    def train(self, df, **kwargs):
        X, self.vectorizer = fit_tfidf(df['comment_text'])  # ← fixed

        y = df[LABEL_COLUMNS].values

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

        self.model.fit(X_train, y_train)
        print(f"✅ TFIDFModel trained.")

        metrics = self.evaluate(X_val, y_val)
        print(f"📊 Validation F1: {metrics['macro']['f1']:.4f} | AUC-ROC: {metrics['macro']['roc_auc']:.4f}")
        return metrics

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        return compute_metrics(y_test, y_pred, y_proba)

    def save(self, path: str = MODEL_PATH):
        with open(path, 'wb') as f:
            pickle.dump((self.model, self.vectorizer), f)
        print(f"💾 Model saved at {path}")

    def load(self, path: str = MODEL_PATH):
        with open(path, 'rb') as f:
            self.model, self.vectorizer = pickle.load(f)
        print(f"📦 Model loaded from {path}")