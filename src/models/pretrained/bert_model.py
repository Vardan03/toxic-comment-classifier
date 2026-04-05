import os
import pickle
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import BertModel as HF_BertModel
from transformers import BertTokenizerFast
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.config import LABEL_COLS


# ── WeightedFocalLoss ──────────────────────────────────────────────────────

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


# ── Dataset ────────────────────────────────────────────────────────────────

class CommentDataset(Dataset):
    def __init__(
        self,
        input_ids:        torch.Tensor,
        attention_mask:   torch.Tensor,
        token_type_ids:   torch.Tensor,
        labels:           Optional[torch.Tensor] = None,
    ):
        self.input_ids      = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels         = labels

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids":      self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "token_type_ids": self.token_type_ids[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


# ── BERTModel ──────────────────────────────────────────────────────────────

class BERTModel(nn.Module):
    """
    BERT-based toxic comment classifier for Jigsaw.

    Изменения относительно исходной версии
    ----------------------------------------
    1. unfreeze_last_n (default=2)
       Последние N трансформер-блоков размораживаются перед обучением.
       Это даёт модели возможность адаптировать контекстные представления
       под конкретную задачу, а не только обучать линейную голову.

    2. Discriminative learning rates
       - Голова (fc):          lr        (e.g. 2e-5)
       - Размороженные блоки:  lr * 0.1  (e.g. 2e-6)
       Стандартная практика fine-tuning: BERT уже хорошо обучен,
       его "подталкивают" осторожно, чтобы не сломать претренированные веса.

    3. batch_size=16 + accumulation_steps=2
       Эффективный batch = 16 * 2 = 32, но VRAM используется как для 16.
       Критично для RTX 3050 6GB: размороженные слои добавляют ~1.5GB
       на градиенты + AdamW optimizer states (2 момента на параметр).

    4. pooler_output вместо last_hidden_state[:, 0, :]
       pooler_output = Linear(Tanh(cls_hidden)) — обучался специально
       для downstream classification. При частично замороженном BERT
       даёт лучшее представление, чем raw [CLS] hidden state.

    5. pos_weight clip(max=100) вместо max=40
       severe_toxic реальный neg/pos ≈ 320. Clip на 40 занижал вес
       и модель всё равно смещалась в сторону positives.
       Focal loss справляется с экстремальными весами — clip можно убрать
       совсем, но 100 — разумный компромисс.

    6. Threshold search расширен до [0.1, 0.90] с шагом 0.02
       При recall~0.99 оптимальный threshold был за пределами [0.1, 0.61].

    Parameters
    ----------
    model_name         : HuggingFace model id
    max_seq_len        : max tokens (≤512 для BERT)
    num_classes        : выходных лейблов (6 для Jigsaw)
    dropout            : dropout перед classifier head
    gamma              : focal loss параметр
    lr                 : learning rate для головы
    bert_lr_multiplier : lr для BERT слоёв = lr * multiplier (default 0.1)
    batch_size         : размер мини-батча (16 для RTX 3050 6GB)
    accumulation_steps : gradient accumulation (эффективный batch = batch*steps)
    epochs             : эпох обучения
    warmup_ratio       : доля шагов для linear lr warmup
    freeze_bert        : если True — все BERT параметры заморожены
    unfreeze_last_n    : размораживать последние N трансформер-блоков
                         (игнорируется если freeze_bert=False)
    device             : 'auto' | 'cuda' | 'cpu'
    """

    def __init__(
        self,
        model_name:           str   = "bert-base-uncased",
        max_seq_len:          int   = 128,
        num_classes:          int   = 6,
        dropout:              float = 0.1,
        gamma:                float = 2.0,
        lr:                   float = 2e-5,
        bert_lr_multiplier:   float = 0.1,
        batch_size:           int   = 16,        # ↓ с 32 для RTX 3050 6GB
        accumulation_steps:   int   = 2,         # эффективный batch = 32
        epochs:               int   = 5,
        warmup_ratio:         float = 0.1,
        freeze_bert:          bool  = True,
        unfreeze_last_n:      int   = 2,         # разморозить последние N блоков
        device:               str   = "auto",
    ):
        super().__init__()

        self.model_name           = model_name
        self.max_seq_len          = max_seq_len
        self.num_classes          = num_classes
        self._dropout_p           = dropout
        self.gamma                = gamma
        self.lr                   = lr
        self.bert_lr_multiplier   = bert_lr_multiplier
        self.batch_size           = batch_size
        self.accumulation_steps   = accumulation_steps
        self.epochs               = epochs
        self.warmup_ratio         = warmup_ratio
        self.freeze_bert          = freeze_bert
        self.unfreeze_last_n      = unfreeze_last_n

        if device == "auto":
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[BERTModel] auto-selected device: {self._device}")
        else:
            self._device = torch.device(device)

        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.bert      = HF_BertModel.from_pretrained(model_name)

        # сначала замораживаем всё
        for param in self.bert.parameters():
            param.requires_grad = False

        # затем размораживаем последние N блоков (если freeze_bert=True)
        # если freeze_bert=False — размораживаем всё ниже
        if not freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = True

        hidden_size = self.bert.config.hidden_size  # 768 для bert-base

        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden_size, num_classes)

        self.criterion:       Optional[WeightedFocalLoss] = None
        self.history:         List[Dict]                  = []
        self._best_threshold: float                       = 0.5

    # ── forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # pooler_output = Linear(Tanh(cls_hidden)) — лучше для частично frozen BERT
        # обучался специально для downstream classification задач
        pooled = outputs.pooler_output     # (B, 768)
        out    = self.drop(pooled)
        return self.fc(out)                # (B, num_classes) — logits

    # ── private helpers ───────────────────────────────────────────────────

    def _get_bert_unfrozen_params(self) -> List[nn.Parameter]:
        """
        Размораживает последние unfreeze_last_n трансформер-блоков BERT
        и возвращает их параметры для discriminative lr optimizer.
        """
        if not self.freeze_bert:
            # всё уже разморожено в __init__
            return list(self.bert.parameters())

        unfrozen = []
        n_layers = len(self.bert.encoder.layer)  # 12 для bert-base

        for i, layer in enumerate(self.bert.encoder.layer):
            if i >= n_layers - self.unfreeze_last_n:
                for param in layer.parameters():
                    param.requires_grad = True
                unfrozen.extend(layer.parameters())

        print(
            f"[BERTModel] разморожено блоков: {self.unfreeze_last_n}/{n_layers} "
            f"(слои {n_layers - self.unfreeze_last_n}–{n_layers - 1})"
        )
        return unfrozen

    def _tokenize_batch(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        enc = self.tokenizer(
            texts,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_token_type_ids=True,
        )
        return enc

    def _make_loader(
        self,
        texts:   List[str],
        labels:  Optional[np.ndarray] = None,
        shuffle: bool = False,
    ) -> DataLoader:
        texts = [str(t) for t in texts]
        enc   = self._tokenize_batch(texts)
        label_tensor = (
            torch.tensor(labels, dtype=torch.float32) if labels is not None else None
        )
        dataset = CommentDataset(
            enc["input_ids"],
            enc["attention_mask"],
            enc["token_type_ids"],
            label_tensor,
        )
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    @staticmethod
    def _compute_pos_weight(y: np.ndarray) -> torch.Tensor:
        pos     = y.sum(axis=0).clip(min=1)
        neg     = (1 - y).sum(axis=0)
        # ↑ clip(max=100) вместо 40 — severe_toxic реальный neg/pos ≈ 320
        weights = (neg / pos).clip(max=100)
        print("[BERTModel] pos_weight:")
        for col, w in zip(LABEL_COLS, weights):
            print(f"  {col:<20} neg/pos = {w:.1f}")
        return torch.tensor(weights, dtype=torch.float32)

    def _find_best_threshold(
        self, probs: np.ndarray, y_true: np.ndarray
    ) -> float:
        """
        Search best threshold в [0.1, 0.90] по macro F1.
        Расширен с 0.61 до 0.90 — при recall~0.99 оптимум был за границей.
        """
        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.1, 0.91, 0.02):   # ← было до 0.61 с шагом 0.05
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
    ) -> "BERTModel":
        self.to(self._device)

        # ── разморозить нужные блоки и собрать param groups ───────────────
        bert_params = self._get_bert_unfrozen_params()

        n_params    = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[BERTModel] total params     = {n_params:,}")
        print(f"[BERTModel] trainable params = {n_trainable:,}")
        print(f"[BERTModel] effective batch  = {self.batch_size * self.accumulation_steps}")

        # ── loss ──────────────────────────────────────────────────────────
        pw = self._compute_pos_weight(y_train).to(self._device)
        self.criterion = WeightedFocalLoss(pos_weight=pw, gamma=self.gamma)

        loader = self._make_loader(X_train, y_train, shuffle=True)

        # ── discriminative learning rates ─────────────────────────────────
        # Голова учится с полным lr, BERT блоки — с уменьшенным lr * multiplier.
        # Иначе большой lr перезапишет хорошие претренированные веса.
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": self.fc.parameters(),
                    "lr": self.lr,
                },
                {
                    "params": bert_params,
                    "lr": self.lr * self.bert_lr_multiplier,   # e.g. 2e-6
                },
            ],
            weight_decay=0.01,
        )

        # total_steps считается по числу реальных optimizer.step() вызовов
        # (каждые accumulation_steps батчей), а не по числу батчей
        effective_steps = (len(loader) + self.accumulation_steps - 1) // self.accumulation_steps
        total_steps     = effective_steps * self.epochs
        warmup_steps    = int(total_steps * self.warmup_ratio)

        from transformers import get_linear_schedule_with_warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        for epoch in range(1, self.epochs + 1):
            self.train()
            total_loss    = 0.0
            optimizer.zero_grad()

            for step, batch in enumerate(loader):
                input_ids      = batch["input_ids"].to(self._device)
                attention_mask = batch["attention_mask"].to(self._device)
                token_type_ids = batch["token_type_ids"].to(self._device)
                labels         = batch["labels"].to(self._device)

                logits = self(input_ids, attention_mask, token_type_ids)

                # делим loss на accumulation_steps чтобы градиенты усреднялись,
                # а не суммировались — иначе эффективный lr вырастет в N раз
                loss = self.criterion(logits, labels) / self.accumulation_steps
                loss.backward()

                total_loss += loss.item() * self.accumulation_steps * len(input_ids)

                is_last_step = (step + 1 == len(loader))
                if (step + 1) % self.accumulation_steps == 0 or is_last_step:
                    nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            avg_loss = total_loss / len(X_train)
            record   = {"epoch": epoch, "train_loss": avg_loss}

            if X_val is not None and y_val is not None:
                val_m = self.evaluate(X_val, y_val, threshold=self._best_threshold)
                record.update({f"val_{k}": v for k, v in val_m.items()})

                probs = self.predict_proba(X_val)
                self._best_threshold = self._find_best_threshold(probs, y_val)

                lr_head = optimizer.param_groups[0]["lr"]
                lr_bert = optimizer.param_groups[1]["lr"]
                print(
                    f"  epoch {epoch:>2}/{self.epochs}  "
                    f"loss={avg_loss:.4f}  "
                    f"val_auc={val_m['roc_auc']:.4f}  "
                    f"val_f1={val_m['f1_macro']:.4f}  "
                    f"thresh={self._best_threshold:.2f}  "
                    f"lr_head={lr_head:.2e}  lr_bert={lr_bert:.2e}"
                )
            else:
                print(f"  epoch {epoch:>2}/{self.epochs}  loss={avg_loss:.4f}")

            self.history.append(record)

        print("[BERTModel] finish.")
        return self

    # ── predict / evaluate / save / load (без изменений) ──────────────────

    def predict_proba(
        self, X: List[str], inference_batch_size: int = 64
    ) -> np.ndarray:
        X = [str(t) for t in X]
        self.eval()
        all_probs: List[np.ndarray] = []

        with torch.no_grad():
            for i in range(0, len(X), inference_batch_size):
                batch_texts = X[i : i + inference_batch_size]
                enc = self.tokenizer(
                    batch_texts,
                    max_length=self.max_seq_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                    return_token_type_ids=True,
                )
                input_ids      = enc["input_ids"].to(self._device)
                attention_mask = enc["attention_mask"].to(self._device)
                token_type_ids = enc["token_type_ids"].to(self._device)

                logits = self(input_ids, attention_mask, token_type_ids)
                probs  = torch.sigmoid(logits).cpu().numpy()
                all_probs.append(probs)

        return np.vstack(all_probs)

    def predict(
        self, X: List[str], threshold: float, inference_batch_size: int = 64
    ) -> np.ndarray:
        return (
            self.predict_proba(X, inference_batch_size) >= threshold
        ).astype(np.int32)

    def evaluate(
        self,
        X_test:               List[str],
        y_test:               np.ndarray,
        threshold:            float,
        inference_batch_size: int = 64,
    ) -> Dict:
        probs = self.predict_proba(X_test, inference_batch_size)
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
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(path, "weights_bert.pt"))
        self.tokenizer.save_pretrained(os.path.join(path, "tokenizer_bert"))

        config = {k: getattr(self, k) for k in (
            "model_name", "max_seq_len", "num_classes", "_dropout_p",
            "gamma", "lr", "bert_lr_multiplier", "batch_size", "accumulation_steps",
            "epochs", "warmup_ratio", "freeze_bert", "unfreeze_last_n",
        )}
        with open(os.path.join(path, "config_bert.pkl"), "wb") as f:
            pickle.dump(config, f)

        print(f"[BERTModel] saved → {path}")

    def load(self, path: str) -> "BERTModel":
        with open(os.path.join(path, "config_bert.pkl"), "rb") as f:
            for k, v in pickle.load(f).items():
                setattr(self, k, v)

        self.tokenizer = BertTokenizerFast.from_pretrained(
            os.path.join(path, "tokenizer_bert")
        )
        self.to(self._device)
        state = torch.load(
            os.path.join(path, "weights_bert.pt"), map_location=self._device
        )
        state = {k: v for k, v in state.items() if not k.startswith("criterion.")}
        self.load_state_dict(state, strict=False)
        self.eval()
        print(f"[BERTModel] loaded ← {path}")
        return self

    def __repr__(self) -> str:
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"BERTModel("
            f"model={self.model_name}, "
            f"max_len={self.max_seq_len}, "
            f"freeze={self.freeze_bert}, "
            f"unfreeze_last_n={self.unfreeze_last_n}, "
            f"eff_batch={self.batch_size * self.accumulation_steps}, "
            f"trainable={n_trainable:,}, "
            f"device={self._device})"
        )