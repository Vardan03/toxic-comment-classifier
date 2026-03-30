import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModel, AutoTokenizer

from src.config import LABEL_COLS


class BertCommentDataset(Dataset):
    def __init__(self, encodings: Dict[str, torch.Tensor], labels: Optional[np.ndarray] = None):
        self.encodings = encodings
        self.labels = (
            torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        )

    def __len__(self) -> int:
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        batch = {
            "input_ids": self.encodings["input_ids"][idx],
            "attention_mask": self.encodings["attention_mask"][idx],
        }
        if self.labels is not None:
            batch["labels"] = self.labels[idx]
        return batch


class BERTClassifier(nn.Module):
    def __init__(
        self,
        pretrained_model_name: str = "bert-base-uncased",
        max_seq_len: int = 256,
        hidden_dim: int = 768,
        num_classes: int = 6,
        dropout: float = 0.3,
        lr: float = 2e-5,
        batch_size: int = 16,
        epochs: int = 3,
        device: str = "auto",
    ):
        super().__init__()

        self.pretrained_model_name = pretrained_model_name
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.lr = lr
        self.batch_size = batch_size
        self.epochs = epochs
        self._dropout_p = dropout

        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[BERTClassifier] auto-selected device: {self._device}")
        else:
            self._device = torch.device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)
        self.bert = AutoModel.from_pretrained(self.pretrained_model_name)
        self.hidden_dim = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)

        self.history: List[Dict] = []
        self._best_threshold = 0.5

    @staticmethod
    def _compute_pos_weight(y: np.ndarray) -> torch.Tensor:
        pos = y.sum(axis=0).clip(min=1)
        neg = (1 - y).sum(axis=0)
        weights = (neg / pos).clip(max=320)
        print("[BERTClassifier] pos_weight:")
        for col, w in zip(LABEL_COLS, weights):
            print(f"  {col:<20} neg/pos = {w:.1f}")
        return torch.tensor(weights, dtype=torch.float32)

    @staticmethod
    def _find_best_threshold(probs: np.ndarray, y_true: np.ndarray) -> float:
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.1, 0.61, 0.05):
            preds = (probs >= t).astype(int)
            f1 = f1_score(y_true, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        return best_t

    def _tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        normalized_texts = [str(text) for text in texts]
        return self.tokenizer(
            normalized_texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors="pt",
        )

    def _loader(
        self,
        texts: List[str],
        labels: Optional[np.ndarray] = None,
        shuffle: bool = False,
    ) -> DataLoader:
        encodings = self._tokenize(texts)
        return DataLoader(
            BertCommentDataset(encodings, labels),
            batch_size=self.batch_size,
            shuffle=shuffle,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        if pooled_output is None:
            pooled_output = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.dropout(pooled_output))
        if return_logits:
            return logits
        return torch.sigmoid(logits)

    def fit(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val: Optional[List[str]] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "BERTClassifier":
        self.to(self._device)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[BERTClassifier] params = {n_params:,}")

        pos_weight = self._compute_pos_weight(y_train).to(self._device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        loader = self._loader(X_train, y_train, shuffle=True)

        for epoch in range(1, self.epochs + 1):
            self.train()
            total_loss = 0.0

            for batch in loader:
                input_ids = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                labels = batch["labels"].to(self._device)

                optimizer.zero_grad()
                logits = self(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_logits=True,
                )
                loss = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * input_ids.size(0)

            avg_loss = total_loss / len(X_train)
            record = {"epoch": epoch, "train_loss": avg_loss}

            if X_val is not None and y_val is not None:
                val_m = self.evaluate(X_val, y_val, threshold=self._best_threshold)
                probs = self.predict_proba(X_val)
                self._best_threshold = self._find_best_threshold(probs, y_val)
                record.update({f"val_{k}": v for k, v in val_m.items()})
                print(
                    f"  epoch {epoch:>2}/{self.epochs}  "
                    f"loss={avg_loss:.4f}  "
                    f"val_auc={val_m['roc_auc']:.4f}  "
                    f"val_f1={val_m['f1_macro']:.4f}  "
                    f"thresh={self._best_threshold:.2f}"
                )
            else:
                print(f"  epoch {epoch:>2}/{self.epochs}  loss={avg_loss:.4f}")

            self.history.append(record)

        print("[BERTClassifier] finish.")
        return self

    def predict_proba(self, X: List[str]) -> np.ndarray:
        self.eval()
        all_probs: List[np.ndarray] = []
        with torch.no_grad():
            for batch in self._loader(X):
                input_ids = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                probs = self(input_ids=input_ids, attention_mask=attention_mask)
                all_probs.append(probs.cpu().numpy())
        return np.vstack(all_probs)

    def predict(self, X: List[str], threshold: Optional[float] = None) -> np.ndarray:
        threshold = self._best_threshold if threshold is None else threshold
        return (self.predict_proba(X) >= threshold).astype(np.int32)

    def evaluate(
        self,
        X_test: List[str],
        y_test: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict:
        threshold = self._best_threshold if threshold is None else threshold
        probs = self.predict_proba(X_test)
        preds = (probs >= threshold).astype(np.int32)

        metrics: Dict = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1_macro": float(f1_score(y_test, preds, average="macro", zero_division=0)),
            "f1_micro": float(f1_score(y_test, preds, average="micro", zero_division=0)),
        }
        try:
            per = roc_auc_score(y_test, probs, average=None)
            metrics["roc_auc"] = float(np.mean(per))
            metrics["per_label_auc"] = {
                LABEL_COLS[i]: float(per[i]) for i in range(len(LABEL_COLS))
            }
        except ValueError:
            metrics["roc_auc"] = float("nan")
            metrics["per_label_auc"] = {}
        return metrics

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "weights_bert.pt"))
        self.tokenizer.save_pretrained(os.path.join(path, "bert_tokenizer"))
        self.bert.config.save_pretrained(os.path.join(path, "bert_backbone_config"))

        config = {
            "model": "bert",
            "pretrained_model_name": self.pretrained_model_name,
            "max_seq_len": self.max_seq_len,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "_dropout_p": self._dropout_p,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "best_threshold": self._best_threshold,
        }
        with open(os.path.join(path, "config.pkl"), "wb") as f:
            pickle.dump(config, f)
        print(f"[BERTClassifier] saved -> {path}")

    def load(self, path: str) -> "BERTClassifier":
        config_path = os.path.join(path, "config.pkl")
        if os.path.exists(config_path):
            with open(config_path, "rb") as f:
                config = pickle.load(f)
            self.pretrained_model_name = config.get(
                "pretrained_model_name", self.pretrained_model_name
            )
            self.max_seq_len = config.get("max_seq_len", self.max_seq_len)
            self.num_classes = config.get("num_classes", self.num_classes)
            self.lr = config.get("lr", self.lr)
            self.batch_size = config.get("batch_size", self.batch_size)
            self.epochs = config.get("epochs", self.epochs)
            self._dropout_p = config.get("_dropout_p", self._dropout_p)
            self._best_threshold = config.get("best_threshold", self._best_threshold)

        tokenizer_path = os.path.join(path, "bert_tokenizer")
        backbone_config_path = os.path.join(path, "bert_backbone_config")
        if os.path.isdir(tokenizer_path):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name)

        if os.path.isdir(backbone_config_path):
            backbone_config = AutoConfig.from_pretrained(backbone_config_path)
            self.bert = AutoModel.from_config(backbone_config)
        else:
            self.bert = AutoModel.from_pretrained(self.pretrained_model_name)
        self.hidden_dim = self.bert.config.hidden_size
        self.dropout = nn.Dropout(self._dropout_p)
        self.classifier = nn.Linear(self.hidden_dim, self.num_classes)

        self.to(self._device)
        self.load_state_dict(
            torch.load(os.path.join(path, "weights_bert.pt"), map_location=self._device)
        )
        self.eval()
        print(f"[BERTClassifier] loaded <- {path}")
        return self

    def __repr__(self) -> str:
        return (
            f"BERTClassifier("
            f"backbone={self.pretrained_model_name}, "
            f"max_seq_len={self.max_seq_len}, "
            f"classes={self.num_classes}, "
            f"device={self._device})"
        )
