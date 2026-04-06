import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from src.utils.metrics import compute_metrics
from src.features.tfidf import fit_tfidf, transform_tfidf
from src.config import TFIDF_PATH, RANDOM_STATE, TEST_SIZE
import os
from src.config import TFIDF_PATH, RANDOM_STATE, TEST_SIZE, LABEL_COLS


class TFIDFModel:
    def __init__(self):
        self.model = OneVsRestClassifier(
            LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight="balanced")
        )
        self.vectorizer = None

    def train(self, df, **kwargs):
        X, self.vectorizer = fit_tfidf(df['comment_text'])

        y = df[LABEL_COLS].values

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

    def save(self, path: str = TFIDF_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump((self.model, self.vectorizer), f)
        print(f"💾 TFIDFModel saved at {path}")

    def load(self, path: str = TFIDF_PATH):
        with open(path, 'rb') as f:
            self.model, self.vectorizer = pickle.load(f)
        print(f"📦 TFIDFModel loaded from {path}")