import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Model as HF_GPT2Model
from transformers import GPT2Tokenizer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, recall_score
from typing import Dict, List, Optional

from src.config import LABEL_COLS

# ─────────────────────────────────────────────
# LOSS FUNCTION
# ─────────────────────────────────────────────

class WeightedFocalLoss(nn.Module):
    def __init__(self, pos_weight: torch.Tensor, gamma: float = 2.0):
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p      = torch.sigmoid(logits)
        bce    = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t    = p * targets + (1 - p) * (1 - targets)
        focal  = (1 - p_t) ** self.gamma
        weight = targets * self.pos_weight + (1 - targets)
        return (weight * focal * bce).mean()


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class CommentDataset(Dataset):
    def __init__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: Optional[torch.Tensor] = None):
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


# ─────────────────────────────────────────────
# GPT-2 MODEL
# ─────────────────────────────────────────────

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
        accumulation_steps: int = 4,
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
        self.accumulation_steps = accumulation_steps
        self.epochs      = epochs
        self.freeze_gpt2 = freeze_gpt2

        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self._device = torch.device(device)

        # Tokenizer Setup
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Backbone Setup
        self.gpt2 = HF_GPT2Model.from_pretrained(model_name)
        self.gpt2.config.pad_token_id = self.tokenizer.eos_token_id

        if freeze_gpt2:
            for param in self.gpt2.parameters():
                param.requires_grad = False

        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(self.gpt2.config.hidden_size, num_classes)
        
        self._best_threshold = 0.5
        self.history = []

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        hidden  = outputs.last_hidden_state
        
        seq_lens = attention_mask.sum(dim=1) - 1
        seq_lens = seq_lens.clamp(min=0)
        idx      = seq_lens.view(-1, 1, 1).expand(-1, 1, hidden.size(-1))
        last_h   = hidden.gather(1, idx).squeeze(1)
        
        return self.fc(self.drop(last_h))

    def _make_loader(self, texts, labels=None, shuffle=False):
        texts = [str(t) for t in texts]
        enc = self.tokenizer(
            texts, max_length=self.max_seq_len, padding="max_length", 
            truncation=True, return_tensors="pt"
        )
        label_tensor = torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        return DataLoader(
            CommentDataset(enc["input_ids"], enc["attention_mask"], label_tensor),
            batch_size=self.batch_size, shuffle=shuffle
        )

    @staticmethod
    def _compute_pos_weight(y: np.ndarray) -> torch.Tensor:
        pos = y.sum(axis=0).clip(min=1)
        neg = (1 - y).sum(axis=0)
        weights = (neg / pos).clip(max=50) 
        return torch.tensor(weights, dtype=torch.float32)

    def _find_best_threshold(self, probs, y_true) -> float:
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.05, 0.95, 0.01):
            preds  = (probs >= t).astype(int)
            recall = recall_score(y_true, preds, average="macro", zero_division=0)
            f1     = f1_score(y_true, preds, average="macro", zero_division=0)
            if recall >= 0.85 and f1 > best_f1:
                best_f1, best_t = f1, float(t)
        return best_t

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.to(self._device)
        pw = self._compute_pos_weight(y_train).to(self._device)
        self.criterion = WeightedFocalLoss(pos_weight=pw, gamma=self.gamma)
        
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        loader    = self._make_loader(X_train, y_train, shuffle=True)

        for epoch in range(1, self.epochs + 1):
            self.train()
            total_loss = 0.0
            optimizer.zero_grad()

            for i, batch in enumerate(loader):
                input_ids = batch["input_ids"].to(self._device)
                mask      = batch["attention_mask"].to(self._device)
                labels    = batch["labels"].to(self._device)

                logits = self(input_ids, mask)
                loss   = self.criterion(logits, labels) / self.accumulation_steps
                loss.backward()

                if (i + 1) % self.accumulation_steps == 0:
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * self.accumulation_steps * len(input_ids)

            if X_val is not None:
                val_m = self.evaluate(X_val, y_val, threshold=self._best_threshold)
                probs = self.predict_proba(X_val)
                self._best_threshold = self._find_best_threshold(probs, y_val)
                
                print(f"Epoch {epoch} | Loss: {total_loss/len(X_train):.4f} | "
                      f"Val AUC: {val_m['roc_auc']:.4f} | Val Recall: {val_m['recall_macro']:.4f} | "
                      f"Thresh: {self._best_threshold:.2f}")

        return self

    def predict_proba(self, X):
        self.eval()
        self.to(self._device)
        probs_list = []
        with torch.no_grad():
            for batch in self._make_loader(X):
                logits = self(batch["input_ids"].to(self._device), batch["attention_mask"].to(self._device))
                probs_list.append(torch.sigmoid(logits).cpu().numpy())
        return np.vstack(probs_list)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(np.int32)

    def evaluate(self, X_test, y_test, threshold=0.5):
        probs = self.predict_proba(X_test)
        preds = (probs >= threshold).astype(np.int32)
        metrics = {
            "accuracy":     float(accuracy_score(y_test, preds)),
            "f1_macro":     float(f1_score(y_test, preds, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, preds, average="macro", zero_division=0)),
            "roc_auc":      float(roc_auc_score(y_test, probs, average="macro"))
        }
        return metrics

    # ─────────────────────────────────────────────
    # RESTORED SAVE & LOAD
    # ─────────────────────────────────────────────

    def save(self, path):
        path = str(path)
        os.makedirs(path, exist_ok=True)
        # Save weights
        torch.save(self.state_dict(), os.path.join(path, "weights_gpt2.pt"))
        # Save tokenizer
        self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))
        # Save config & best threshold
        config = {
            "model_name":  self.model_name,
            "max_seq_len": self.max_seq_len,
            "num_classes": self.num_classes,
            "dropout":     self._dropout_p,
            "best_threshold": self._best_threshold
        }
        with open(os.path.join(path, "config_gpt2.pkl"), "wb") as f:
            pickle.dump(config, f)

    def load(self, path):
        with open(os.path.join(path, "config_gpt2.pkl"), "rb") as f:
            config = pickle.load(f)
            self.model_name      = config.get("model_name", "gpt2")
            self.max_seq_len     = config.get("max_seq_len", 128)
            self.num_classes     = config.get("num_classes", 6)
            self._best_threshold = config.get("best_threshold", 0.5)

        # 2. Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(os.path.join(path, "tokenizer"))
        
        # 3. Load state dict
        weights_path = os.path.join(path, "weights_gpt2.pt")
        
        # ADD strict=False HERE ⬇️
        self.load_state_dict(
            torch.load(weights_path, map_location=self._device), 
            strict=False
        )
        
        self.to(self._device)
        self.eval()