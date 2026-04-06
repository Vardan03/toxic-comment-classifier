import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, recall_score
from src.config import LABEL_COLS


def compute_metrics(y_true, y_pred, y_pred_proba):
    y_proba = np.array(y_pred_proba)

    results = {}

    for i, label in enumerate(LABEL_COLS):
        auc    = roc_auc_score(y_true[:, i], y_proba[:, i])
        f1     = f1_score(y_true[:, i], y_pred[:, i])
        recall = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        results[label] = {"roc_auc": auc, "f1": f1, "recall": recall}

    results["macro"] = {
        "roc_auc": roc_auc_score(y_true, y_proba, average='macro'),
        "f1":      f1_score(y_true, y_pred, average='macro'),
        "recall":  recall_score(y_true, y_pred, average='macro', zero_division=0)
    }

    return results

def print_metrics(results):
    print("=" * 55)
    print(f"{'Label':<20} {'ROC AUC':>10} {'F1':>10} {'Recall':>10}")
    print("=" * 55)

    for label, metrics in results.items():
        if label == "macro":
            continue
        print(f"{label:<20} {metrics['roc_auc']:>10.4f} {metrics['f1']:>10.4f} {metrics['recall']:>10.4f}")

    print("=" * 55)
    print(f"{'MACRO AVG':<20} {results['macro']['roc_auc']:>10.4f} {results['macro']['f1']:>10.4f} {results['macro']['recall']:>10.4f}")