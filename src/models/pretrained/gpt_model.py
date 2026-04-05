import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Model as HF_GPT2Model
from transformers import GPT2Tokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch.nn.functional as F
from src.config import LABEL_COLS


class WeightedFocalLoss(nn.Module):
    """
    formula (per element)
    ---------------------
    p   = sigmoid(logit)
    p_t = p if y=1,  (1-p) if y=0
    FL  = -pos_weight * y * (1-p_t)^γ * log(p_t)
          - (1-y) * (1-p_t)^γ * log(1-p_t)

    pos_weight : for imbalanced data (neg/pos per label), clipped at 320
    gamma      : focus on hard examples (γ=2.0 default)
    """

    def __init__(self, pos_weight: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p     = torch.sigmoid(logits)
        bce   = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t   = p * targets + (1 - p) * (1 - targets)
        focal = (1 - p_t) ** self.gamma
        weight = targets * self.pos_weight + (1 - targets)
        return (weight * focal * bce).mean()


# ── Dataset ────────────────────────────────────────────────────────────────

class CommentDataset(Dataset):
    """Holds pre-tokenized input_ids + attention_mask tensors."""

    def __init__(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        labels:         Optional[torch.Tensor] = None,
    ):
        self.input_ids      = input_ids
        self.attention_mask = attention_mask
        self.labels         = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


# ── GPT2Model ──────────────────────────────────────────────────────────────

class GPT2Model(nn.Module):
    def __init__(
        self,
        model_name:  str   = "gpt2",
        max_seq_len: int   = 128,
        num_classes: int   = 6,
        dropout:     float = 0.1,
        gamma:       float = 2.0,
        lr:          float = 2e-5,
        batch_size:  int   = 16,
        epochs:      int   = 5,
        freeze_gpt2: bool  = True,
        device:      str   = "auto",
    ):
        super().__init__()

        self.model_name  = model_name
        self.max_seq_len = max_seq_len
        self.num_classes = num_classes
        self._dropout_p  = dropout
        self.gamma       = gamma
        self.lr          = lr
        self.batch_size  = batch_size
        self.epochs      = epochs
        self.freeze_gpt2 = freeze_gpt2

        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[GPT2Model] auto-selected device: {self._device}")
        else: 
            self._device = torch.device(device)

        # ── tokenizer ─────────────────────────────────────────────────────
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # GPT-2 has no pad token by default — use eos as pad
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── GPT-2 backbone ─────────────────────────────────────────────────
        self.gpt2 = HF_GPT2Model.from_pretrained(model_name)
        self.gpt2.config.pad_token_id = self.tokenizer.eos_token_id

        if freeze_gpt2:
            for param in self.gpt2.parameters():
                param.requires_grad = False

        hidden_size = self.gpt2.config.hidden_size   # 768 for gpt2

        # ── classifier head ────────────────────────────────────────────────
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size, num_classes)

        # ── state ──────────────────────────────────────────────────────────
        self.criterion: Optional[WeightedFocalLoss] = None
        self.history:   List[Dict]                  = []
        self._best_threshold: float                 = 0.5

    # ── forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        input_ids, attention_mask : (B, T)
        returns logits             : (B, num_classes)
        """
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        hidden  = outputs.last_hidden_state           # (B, T, H)

        # index of last non-pad token per sample
        seq_lens  = attention_mask.sum(dim=1) - 1     # (B,)
        seq_lens  = seq_lens.clamp(min=0)
        # gather last real token: hidden[b, seq_lens[b], :]
        idx       = seq_lens.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, hidden.size(-1))
        last_h    = hidden.gather(1, idx).squeeze(1)  # (B, H)

        out = self.drop(last_h)
        return self.fc(out)                            # (B, num_classes) — logits

    # ── private helpers ───────────────────────────────────────────────────

    def _tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize a list of texts → dict with input_ids & attention_mask."""
        enc = self.tokenizer(
            texts,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return enc  # {"input_ids": (N,T), "attention_mask": (N,T)}

    def _make_loader(
        self,
        texts:   List[str],
        labels:  Optional[np.ndarray] = None,
        shuffle: bool = False,
    ) -> DataLoader:
        enc = self._tokenize_batch(texts)
        label_tensor = (
            torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        )
        dataset = CommentDataset(
            enc["input_ids"],
            enc["attention_mask"],
            label_tensor,
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    @staticmethod
    def _compute_pos_weight(y: np.ndarray) -> torch.Tensor:
        pos     = y.sum(axis=0).clip(min=1)
        neg     = (1 - y).sum(axis=0)
        weights = (neg / pos).clip(max=320)
        print("[GPT2Model] pos_weight:")
        for col, w in zip(LABEL_COLS, weights):
            print(f"  {col:<20} neg/pos = {w:.1f}")
        return torch.tensor(weights, dtype=torch.float32)

    def _find_best_threshold(
        self, probs: np.ndarray, y_true: np.ndarray
    ) -> float:
        """Search best threshold in [0.1, 0.6] by macro F1."""
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.1, 0.61, 0.05):
            preds = (probs >= t).astype(int)
            f1    = f1_score(y_true, preds, average="macro", zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)
        return best_t

    # ── public API ────────────────────────────────────────────────────────

    def fit(
        self,
        X_train: List[str],
        y_train: np.ndarray,
        X_val:   Optional[List[str]]  = None,
        y_val:   Optional[np.ndarray] = None,
    ) -> "GPT2Model":
        """
        Fine-tune GPT-2 on training data.

        Parameters
        ----------
        X_train : raw comment strings
        y_train : float32 array (N, 6)
        X_val   : optional validation texts
        y_val   : optional validation labels — enables per-epoch metrics
                  and auto threshold selection
        """
        self.to(self._device)

        n_params     = sum(p.numel() for p in self.parameters())
        n_trainable  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[GPT2Model] total params     = {n_params:,}")
        print(f"[GPT2Model] trainable params = {n_trainable:,}")
        print(f"[GPT2Model] freeze_gpt2      = {self.freeze_gpt2}")

        # loss
        pw = self._compute_pos_weight(y_train).to(self._device)
        self.criterion = WeightedFocalLoss(pos_weight=pw, gamma=self.gamma)

        loader    = self._make_loader(X_train, y_train, shuffle=True)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=0.01,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=1, factor=0.5
        )

        for epoch in range(1, self.epochs + 1):
            self.train()
            total_loss = 0.0

            for batch in loader:
                input_ids      = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                labels         = batch["labels"].to(self._device)

                optimizer.zero_grad()
                logits = self(input_ids, attention_mask)
                loss   = self.criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item() * len(input_ids)

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
                    f"thresh={self._best_threshold:.2f}  "
                    f"lr={lr_now:.2e}"
                )
            else:
                print(f"  epoch {epoch:>2}/{self.epochs}  loss={avg_loss:.4f}")

            self.history.append(record)

        print("[GPT2Model] finish.")
        return self

    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Return probability array (N, 6) — sigmoid of logits."""
        self.eval()
        all_probs: List[np.ndarray] = []
        with torch.no_grad():
            for batch in self._make_loader(X):
                input_ids      = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                logits = self(input_ids, attention_mask)
                probs  = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)
        return np.vstack(all_probs)

    def predict(self, X: List[str], threshold: float) -> np.ndarray:
        """Return binary predictions (N, 6)."""
        return (self.predict_proba(X) >= threshold).astype(np.int32)

    def evaluate(
        self, X_test: List[str], y_test: np.ndarray, threshold: float
    ) -> Dict:
        """Return metrics dict: accuracy, f1_macro, f1_micro, roc_auc, per_label_auc."""
        probs = self.predict_proba(X_test)
        preds = (probs >= threshold).astype(np.int32)

        metrics: Dict = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1_macro": float(f1_score(y_test, preds, average="macro",  zero_division=0)),
            "f1_micro": float(f1_score(y_test, preds, average="micro",  zero_division=0)),
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
        """Save to directory: weights_gpt2.pt, tokenizer/, config.pkl"""
        os.makedirs(path, exist_ok=True)

        # save classifier head + any fine-tuned gpt2 weights
        torch.save(self.state_dict(), os.path.join(path, "weights_gpt2.pt"))

        # save tokenizer (handles vocab, special tokens)
        self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))

        config = {k: getattr(self, k) for k in (
            "model_name", "max_seq_len", "num_classes", "_dropout_p",
            "gamma", "lr", "batch_size", "epochs", "freeze_gpt2",
        )}
        with open(os.path.join(path, "config_gpt2.pkl"), "wb") as f:
            pickle.dump(config, f)

        print(f"[GPT2Model] saved → {path}")

    def load(self, path: str) -> "GPT2Model":
        """Restore from directory produced by .save()."""
        with open(os.path.join(path, "config_gpt2.pkl"), "rb") as f:
            for k, v in pickle.load(f).items():
                setattr(self, k, v)

        # reload tokenizer from saved dir
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            os.path.join(path, "tokenizer")
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.to(self._device)
        state = torch.load(
            os.path.join(path, "weights_gpt2.pt"), map_location=self._device
        )
        # drop criterion buffers if present
        state = {k: v for k, v in state.items() if not k.startswith("criterion.")}
        self.load_state_dict(state, strict=False)
        self.eval()
        print(f"[GPT2Model] loaded ← {path}")
        return self

    def __repr__(self) -> str:
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"GPT2Model("
            f"model={self.model_name}, "
            f"max_len={self.max_seq_len}, "
            f"freeze={self.freeze_gpt2}, "
            f"trainable={n_trainable:,}, "
            f"device={self._device})"
        )