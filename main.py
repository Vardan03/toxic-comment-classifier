import argparse
import numpy as np

from src.data.load_data import (
    load_train_data, load_test_data, load_test_labels,
    load_preprocess_train_data, load_preprocess_test_data,
    load_preprocess_test_labels
)
from src.data.preprocess import preprocess_train, preprocess_test
from src.utils.metrics import compute_metrics, print_metrics
from src.utils.logger import save_results
from src.config import LABEL_COLS, PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH, PROCESSED_TEST_LABELS_PATH, THRESHOLD
from predict import predict, predict_proba


# ─────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────

def run_preprocessing():
    print("📂 Loading raw data...")
    train       = load_train_data()
    test        = load_test_data()
    test_labels = load_test_labels()

    print("🔄 Preprocessing...")
    train            = preprocess_train(train)
    test, test_labels = preprocess_test(test, test_labels)

    print("💾 Saving processed data...")
    PROCESSED_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_TRAIN_PATH,             index=False)
    test.to_csv(PROCESSED_TEST_PATH,               index=False)
    test_labels.to_csv(PROCESSED_TEST_LABELS_PATH, index=False)

    print(f"✅ Preprocessing complete!")
    print(f"   Train: {len(train)} comments")
    print(f"   Test:  {len(test)} comments")


# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────

def run_evaluation(models: list, threshold: float):
    print("📂 Loading preprocessed test data...")
    test        = load_preprocess_test_data()
    test_labels = load_preprocess_test_labels()

    X_test = test["comment_text"].tolist()
    y_true = test_labels[LABEL_COLS].values

    for model_name in models:
        print(f"\n📊 Evaluating {model_name} (threshold={threshold})...")
        try:
            y_pred       = predict(X_test, model_name=model_name, threshold=threshold).values
            y_pred_proba = predict_proba(X_test, model_name=model_name)

            metrics = compute_metrics(y_true, y_pred, y_pred_proba)
            print_metrics(metrics)
            save_results(metrics, model_name=f"{model_name}_t{threshold}")
            print(f"✅ {model_name} results saved to reports/result.txt")

        except FileNotFoundError:
            print(f"⚠️  {model_name} model not found — skipping. Train it first with:")
            print(f"   python train.py --model {model_name}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Preprocess data or evaluate trained models")
    parser.add_argument(
        "--mode",
        choices=["preprocess", "evaluate"],
        required=True,
        help="preprocess — clean and save raw data | evaluate — evaluate trained models"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["tfidf", "rnn", "lstm", "bert", "gpt"],
        default=["tfidf", "rnn", "lstm", "bert", "gpt"],
        help="Models to evaluate (default: all). Example: --models tfidf lstm"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help=f"Decision threshold for evaluation (default: {THRESHOLD})"
    )
    args = parser.parse_args()

    if args.mode == "preprocess":
        run_preprocessing()
    elif args.mode == "evaluate":
        run_evaluation(args.models, threshold=args.threshold)


if __name__ == "__main__":
    main()