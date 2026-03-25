from abc import ABC, abstractmethod
import torch


class BaseModel(ABC):
    """
    Abstract base class for all models in the toxic comment classifier.
    Every model (TF-IDF, RNN, LSTM, BERT, GPT) must implement these methods.
    """

    @abstractmethod
    def train(self, X_train, y_train, **kwargs):
        """Train the model on the given data."""
        pass

    @abstractmethod
    def predict(self, X):
        """Return predicted labels for input X."""
        pass

    @abstractmethod
    def predict_proba(self, X):
        """Return prediction probabilities for input X."""
        pass

    @abstractmethod
    def evaluate(self, X_test, y_test):
        """Evaluate the model and return metrics dict."""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save the model to the given path."""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load the model from the given path."""
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__}>"