import os
import pickle
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score

from src.config import LABEL_COLS

class WeightedFocalLoss(nn.Module):
    """
    Weighted Focal Loss for imbalanced multi-label classification.

    Combines two ideas:
      1. pos_weight — makes the model pay more attention to toxic (positive) class
                      calculated as neg/pos ratio per label, so rare labels get
                      higher weight automatically
      2. gamma      — focuses training on hard/misclassified examples by
                      down-weighting easy ones via (1-p_t)^gamma term

    Formula per element:
        p     = sigmoid(logit)
        p_t   = p if y=1, (1-p) if y=0
        FL    = -pos_weight * y * (1-p_t)^γ * log(p_t)
                - (1-y) * (1-p_t)^γ * log(1-p_t)

    Args:
        pos_weight : tensor of shape (num_labels,), neg/pos ratio per label
        gamma      : focusing parameter, default 2.0
    """

    def __init__(self, pos_weight: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        p_t = p * targets + (1 - p) * (1 - targets)
        focal = (1 - p_t) ** self.gamma
        weight = targets * self.pos_weight + (1 - targets)
        return (weight * focal * bce).mean()

class Vocabulary:
    """
    Word-level vocabulary built from training corpus.

    Special tokens:
        index 0 → <PAD>  used for padding sequences to equal length
        index 1 → <UNK>  used for words not seen during training

    Args:
        max_size : maximum vocabulary size, default 30,000
    """

    PAD, UNK = "<PAD>", "<UNK>"

    def __init__(self, max_size: int = 30_000):
        self.max_size = max_size
        self.token2idx: Dict[str, int] = {}
        self.idx2token: Dict[int, str] = {}

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Lowercase and split on non-alphanumeric characters."""
        return re.sub(r"[^a-z0-9\s]", " ", text.lower()).split()

    def fit(self, texts: List[str]) -> "Vocabulary":
        """
        Build vocabulary from list of texts.
        Keeps only the most frequent max_size-2 tokens
        (reserving 2 slots for PAD and UNK).
        """
        cnt: Counter = Counter()
        for t in texts:
            cnt.update(self._tokenize(t))
        self.token2idx = {self.PAD: 0, self.UNK: 1}
        for tok, _ in cnt.most_common(self.max_size - 2):
            self.token2idx[tok] = len(self.token2idx)
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        return self

    def encode(self, text: str, max_len: int) -> List[int]:
        """
        Convert text to list of token indices.
        Truncates to max_len, then right-pads with 0 (PAD).
        Unknown words get index 1 (UNK).
        """
        ids = [self.token2idx.get(t, 1) for t in self._tokenize(text)[:max_len]]
        ids += [0] * (max_len - len(ids))
        return ids

    def __len__(self) -> int:
        return len(self.token2idx)

class CommentDataset(Dataset):
    """
    PyTorch Dataset wrapping encoded sequences and optional labels.

    Args:
        seqs   : np.ndarray of shape (n_samples, max_len), token indices
        labels : np.ndarray of shape (n_samples, num_labels), optional
                 None during inference
    """

    def __init__(self, seqs: np.ndarray, labels: Optional[np.ndarray] = None):
        self.seqs = torch.tensor(seqs, dtype=torch.long)
        self.labels = (
            torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        )

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx):
        """Returns (seq, label) tuple during training, just seq during inference."""
        if self.labels is not None:
            return self.seqs[idx], self.labels[idx]
        return self.seqs[idx]

class LSTMModel(nn.Module):
    """
    Bidirectional LSTM for multi-label toxic comment classification.
    Supports pretrained GloVe embeddings via load_pretrained_embeddings().

    Architecture:
        GloVe Embeddings → Dropout → BiLSTM → Dropout → Linear → (Sigmoid at inference)

    Args:
        vocab_size  : max vocabulary size, default 30,000
        max_seq_len : max tokens per comment, default 200
        embed_dim   : embedding size, must match GloVe dim (e.g. 100), default 100
        hidden_dim  : LSTM hidden size per direction, default 256
                      output is hidden_dim*2 due to bidirectional
        num_layers  : number of stacked LSTM layers, default 3
        num_classes : number of output labels, default 6
        gamma       : focal loss focusing parameter, default 2.0
        dropout     : dropout probability, default 0.5
        lr          : Adam learning rate, default 1e-3
        batch_size  : training batch size, default 64
        epochs      : number of training epochs, default 5
        device      : 'auto', 'cpu', or 'cuda', default 'auto'
    """

    def __init__(
        self,
        vocab_size:  int   = 30_000,
        max_seq_len: int   = 200,
        embed_dim:   int   = 100,
        hidden_dim:  int   = 256,
        num_layers:  int   = 3,
        num_classes: int   = 6,
        gamma:       float = 2.0,
        dropout:     float = 0.5,
        lr:          float = 1e-3,
        batch_size:  int   = 64,
        epochs:      int   = 5,
        device:      str   = "auto",
    ):
        super().__init__()

        self.vocab_size  = vocab_size
        self.max_seq_len = max_seq_len
        self.embed_dim   = embed_dim
        self.hidden_dim  = hidden_dim
        self.num_layers  = num_layers
        self.num_classes = num_classes
        self.lr          = lr
        self.batch_size  = batch_size
        self.gamma       = gamma
        self.epochs      = epochs
        self._dropout_p  = dropout

        self._best_threshold: float = 0.5

        self.embedding = nn.Embedding(1, embed_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[LSTMModel] auto-selected device: {self._device}")
        else:
            self._device = torch.device(device)

        self.vocab:   Optional[Vocabulary] = None
        self.history: List[Dict]           = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass — returns raw logits (no sigmoid).
        Sigmoid applied in predict_proba() to avoid double sigmoid bug.
        """
        emb = self.drop(self.embedding(x))    # (B, T, E)
        _, (h, _) = self.lstm(emb)            # h: (num_layers*2, B, H)
        h = torch.cat([h[-2], h[-1]], dim=1)  # (B, 2*H)
        out = self.drop(h)
        return self.fc(out)                   # (B, C)

    def _reinit_embedding(self):
        """Rebuild embedding with correct vocab size after fit()."""
        old = self.embedding
        self.embedding = nn.Embedding(
            len(self.vocab), self.embed_dim, padding_idx=0
        ).to(self._device)
        with torch.no_grad():
            n = min(old.weight.shape[0], self.embedding.weight.shape[0])
            self.embedding.weight[:n] = old.weight[:n]

    def load_pretrained_embeddings(self, glove_path: str) -> None:
        """
        Load GloVe embeddings into the embedding layer.

        How it works:
          - reads GloVe file line by line
          - for each word found in our vocabulary, replaces random
            weights with the pretrained GloVe vector
          - words not in GloVe keep their random initialization

        Args:
            glove_path : path to GloVe file e.g. 'glove.6B.100d.txt'
                         embed_dim must match the GloVe dimension!
        """
        print(f"[LSTMModel] loading GloVe from {glove_path}...")
        found = 0
        with open(glove_path, encoding='utf-8') as f:
            for line in f:
                parts = line.split()
                word  = parts[0]
                if word in self.vocab.token2idx:
                    idx = self.vocab.token2idx[word]
                    vec = torch.tensor(
                        [float(x) for x in parts[1:]], dtype=torch.float32
                    )
                    self.embedding.weight.data[idx] = vec
                    found += 1
        print(f"[LSTMModel] found {found}/{len(self.vocab)} words in GloVe "
              f"({found/len(self.vocab)*100:.1f}%)")

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Convert list of texts to numpy array of token indices."""
        return np.array(
            [self.vocab.encode(t, self.max_seq_len) for t in texts],
            dtype=np.int64,
        )

    def _loader(
        self,
        seqs:    np.ndarray,
        labels:  Optional[np.ndarray] = None,
        shuffle: bool = False,
    ) -> DataLoader:
        """Create a DataLoader from encoded sequences and optional labels."""
        return DataLoader(
            CommentDataset(seqs, labels),
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

    @staticmethod
    def _compute_pos_weight(y: np.ndarray) -> torch.Tensor:
        """
        Compute per-label positive weights for WeightedFocalLoss.
        Clipped at 50 for better precision/recall balance.
        """
        pos = y.sum(axis=0).clip(min=1)
        neg = (1 - y).sum(axis=0)
        weights = (neg / pos).clip(max=50)
        print("[LSTMModel] pos_weight per label:")
        for col, w in zip(LABEL_COLS, weights):
            print(f"  {col:<20} neg/pos = {w:.1f}")
        return torch.tensor(weights, dtype=torch.float32)

    def _find_best_threshold(self, probs: np.ndarray, y_true: np.ndarray) -> float:
        """
        Find threshold maximizing F1 while keeping recall >= 0.85.
        Fine-grained search [0.05, 0.95] step 0.01.
        """
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.05, 0.95, 0.01):
            preds  = (probs >= t).astype(int)
            recall = recall_score(y_true, preds, average="macro", zero_division=0)
            f1     = f1_score(y_true, preds, average="macro", zero_division=0)
            if recall >= 0.85 and f1 > best_f1:
                best_f1, best_t = f1, float(t)
        return best_t

    def fit(
        self,
        X_train:    List[str],
        y_train:    np.ndarray,
        X_val:      Optional[List[str]]  = None,
        y_val:      Optional[np.ndarray] = None,
        glove_path: Optional[str]        = None,
    ) -> "LSTMModel":
        """
        Train the model.

        Args:
            X_train    : list of training comment texts
            y_train    : np.ndarray of shape (n, 6), binary labels
            X_val      : optional validation texts
            y_val      : optional validation labels
            glove_path : optional path to GloVe file
                         if provided, loads pretrained embeddings before training
        """
        self.vocab = Vocabulary(self.vocab_size).fit(X_train)
        self._reinit_embedding()
        self.to(self._device)

        # load GloVe if provided
        if glove_path is not None:
            self.load_pretrained_embeddings(glove_path)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[LSTMModel] total parameters: {n_params:,}")

        pw = self._compute_pos_weight(y_train).to(self._device)
        self.criterion = WeightedFocalLoss(pos_weight=pw, gamma=self.gamma)

        loader    = self._loader(self._encode(X_train), y_train, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=2, factor=0.5
        )

        for epoch in range(1, self.epochs + 1):
            self.train()
            total_loss = 0.0
            for batch_seqs, batch_labels in loader:
                batch_seqs   = batch_seqs.to(self._device)
                batch_labels = batch_labels.to(self._device)
                optimizer.zero_grad()
                loss = self.criterion(self(batch_seqs), batch_labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 5.0)
                optimizer.step()
                total_loss += loss.item() * len(batch_seqs)

            avg_loss = total_loss / len(X_train)
            record   = {"epoch": epoch, "train_loss": avg_loss}

            if X_val is not None and y_val is not None:
                val_m = self.evaluate(X_val, y_val, threshold=self._best_threshold)
                record.update({f"val_{k}": v for k, v in val_m.items()})
                scheduler.step(val_m["roc_auc"])

                probs = self.predict_proba(X_val)
                self._best_threshold = self._find_best_threshold(probs, y_val)

                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"  epoch {epoch:>2}/{self.epochs}  "
                    f"loss={avg_loss:.4f}  "
                    f"val_auc={val_m['roc_auc']:.4f}  "
                    f"val_f1={val_m['f1_macro']:.4f}  "
                    f"val_recall={val_m['recall_macro']:.4f}  "
                    f"thresh={self._best_threshold:.2f}  "
                    f"lr={lr_now:.2e}"
                )
            else:
                print(f"  epoch {epoch:>2}/{self.epochs}  loss={avg_loss:.4f}")

            self.history.append(record)

        print("[LSTMModel] training complete.")
        return self

    def predict_proba(self, X: List[str]) -> np.ndarray:
        """
        Return predicted probabilities.
        Sigmoid applied here on raw logits from forward().
        """
        if self.vocab is None:
            raise RuntimeError("Model not trained yet — call fit() first.")
        self.eval()
        all_probs: List[np.ndarray] = []
        with torch.no_grad():
            for batch in self._loader(self._encode(X)):
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                probs = torch.sigmoid(self(batch.to(self._device)))
                all_probs.append(probs.cpu().numpy())
        return np.vstack(all_probs)

    def predict(self, X: List[str], threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions. Use _best_threshold for recall-optimized predictions."""
        return (self.predict_proba(X) >= threshold).astype(np.int32)

    def evaluate(
        self,
        X_test:    List[str],
        y_test:    np.ndarray,
        threshold: float = 0.5,
    ) -> Dict:
        """
        Evaluate model — returns accuracy, F1, recall and AUC metrics.
        Recall tracked per label and as macro since it's primary metric.
        """
        probs = self.predict_proba(X_test)
        preds = self.predict(X_test, threshold)

        metrics: Dict = {
            "accuracy":     float(accuracy_score(y_test, preds)),
            "f1_macro":     float(f1_score(y_test, preds, average="macro",  zero_division=0)),
            "f1_micro":     float(f1_score(y_test, preds, average="micro",  zero_division=0)),
            "recall_macro": float(recall_score(y_test, preds, average="macro", zero_division=0)),
            "per_label_recall": {
                LABEL_COLS[i]: float(recall_score(y_test[:, i], preds[:, i], zero_division=0))
                for i in range(len(LABEL_COLS))
            },
        }
        try:
            per = roc_auc_score(y_test, probs, average=None)
            metrics["roc_auc"]       = float(np.mean(per))
            metrics["per_label_auc"] = {
                LABEL_COLS[i]: float(per[i]) for i in range(len(LABEL_COLS))
            }
        except ValueError:
            metrics["roc_auc"]       = float("nan")
            metrics["per_label_auc"] = {}

        return metrics

    def save(self, path: str):
        """
        Save model to directory:
          - weights_lstm.pt : PyTorch state dict
          - vocab_lstm.pkl  : fitted Vocabulary
          - config.pkl      : hyperparameters
        """
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "weights_lstm.pt"))
        with open(os.path.join(path, "vocab_lstm.pkl"), "wb") as f:
            pickle.dump(self.vocab, f)
        config = {k: getattr(self, k) for k in (
            "vocab_size", "max_seq_len", "embed_dim", "hidden_dim",
            "num_layers", "num_classes", "_dropout_p", "lr",
            "batch_size", "epochs"
        )}
        with open(os.path.join(path, "config.pkl"), "wb") as f:
            pickle.dump(config, f)
        print(f"[LSTMModel] saved → {path}")

    def load(self, path: str) -> "LSTMModel":
        """Load model from directory saved by save()."""
        with open(os.path.join(path, "config.pkl"), "rb") as f:
            for k, v in pickle.load(f).items():
                setattr(self, k, v)
        with open(os.path.join(path, "vocab_lstm.pkl"), "rb") as f:
            self.vocab = pickle.load(f)
        self._reinit_embedding()
        self.to(self._device)
        self.load_state_dict(
            torch.load(
                os.path.join(path, "weights_lstm.pt"),
                map_location=self._device
            ),
            strict=False
        )
        self.eval()
        print(f"[LSTMModel] loaded ← {path}")
        return self

    def __repr__(self) -> str:
        fitted = self.vocab is not None
        return (
            f"LSTMModel("
            f"vocab={'fitted' if fitted else 'not fitted'}, "
            f"embed={self.embed_dim}, "
            f"hidden={self.hidden_dim}, "
            f"layers={self.num_layers}, "
            f"device={self._device})"
        )