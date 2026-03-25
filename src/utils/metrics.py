import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

def compute_metrics(y_true, y_pred, y_pred_proba):
    y_proba = np.column_stack([
        proba[:, 1] if proba.ndim == 2 else proba
        for proba in y_pred_proba
    ]).T

    results = {}

    for i, label in enumerate(LABELS):
        auc = roc_auc_score(y_true[:, i], y_proba[:, i])
        f1  = f1_score(y_true[:, i], y_pred[:, i])
        results[label] = {"roc_auc": auc, "f1": f1}

    results["macro"] = {
        "roc_auc": roc_auc_score(y_true, y_proba, average='macro'),
        "f1": f1_score(y_true, y_pred, average='macro')
    }

    return results


def print_metrics(results):
    print("=" * 40)
    print(f"{'Label':<20} {'ROC AUC':>10} {'F1':>10}")
    print("=" * 40)

    for label, metrics in results.items():
        if label == "macro":
            continue
        print(f"{label:<20} {metrics['roc_auc']:>10.4f} {metrics['f1']:>10.4f}")

    print("=" * 40)
    print(f"{'MACRO AVG':<20} {results['macro']['roc_auc']:>10.4f} {results['macro']['f1']:>10.4f}")