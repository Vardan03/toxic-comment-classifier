from datetime import datetime
from src.config import REPORTS_PATH

def save_results(results, model_name="model"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(REPORTS_PATH, "a") as f:  # append!
        f.write(f"\n=== {model_name} | {timestamp} ===\n")
        f.write(f"{'Label':<20} {'ROC AUC':>10} {'F1':>10}\n")
        f.write("=" * 40 + "\n")

        for label, metrics in results.items():
            if label == "macro":
                continue
            f.write(f"{label:<20} {metrics['roc_auc']:>10.4f} {metrics['f1']:>10.4f}\n")

        f.write("=" * 40 + "\n")
        f.write(f"{'MACRO AVG':<20} {results['macro']['roc_auc']:>10.4f} {results['macro']['f1']:>10.4f}\n")