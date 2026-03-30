import argparse

from predict import (
    predict,
    predict_deep_learning,
    predict_deep_learning_proba,
    predict_proba,
)
from src.config import LABEL_COLS, THRESHOLD
from src.data.load_data import (
    load_preprocess_test_data,
    load_preprocess_test_labels,
    load_preprocess_train_data,
)
from src.utils.logger import save_results
from src.utils.metrics import compute_metrics, print_metrics
from train import train_deep_learning_model, train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Toxic comment classifier")
    parser.add_argument(
        "--model",
        choices=["baseline", "rnn", "lstm", "bert"],
        default="rnn",
        help="Model type to train or evaluate.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the selected model before evaluation.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=THRESHOLD,
        help="Prediction threshold for multi-label outputs.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Maximum token length used by BERT.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size override for deep learning training.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Epoch count override for deep learning training.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate override for deep learning training.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    train = load_preprocess_train_data()
    test = load_preprocess_test_data()
    test_labels = load_preprocess_test_labels()

    if args.train:
        if args.model == "baseline":
            train_model(train)
        else:
            train_y = train[LABEL_COLS].to_numpy().astype(np.float32)
            config = {"model": args.model}
            if args.max_seq_len is not None:
                config["max_seq_len"] = args.max_seq_len
            if args.batch_size is not None:
                config["batch_size"] = args.batch_size
            if args.epochs is not None:
                config["epochs"] = args.epochs
            if args.lr is not None:
                config["lr"] = args.lr
            train_deep_learning_model(
                X_train=train["comment_text"].astype(str).tolist(),
                y_train=train_y,
                config=config,
            )

    if args.model == "baseline":
        y_pred = predict(test["comment_text"].astype(str).tolist(), threshold=args.threshold)
        y_pred_proba = predict_proba(test["comment_text"].astype(str).tolist())
        y_pred_array = y_pred.values
    else:
        config = {"model": args.model}
        y_pred_array = predict_deep_learning(
            test["comment_text"].astype(str).tolist(),
            threshold=args.threshold,
            config=config,
        )
        y_pred_proba = predict_deep_learning_proba(
            test["comment_text"].astype(str).tolist(),
            config=config,
        )
        y_pred = y_pred_array

    print(y_pred)

    y_true = test_labels.drop(columns=["id"]).values
    metrics = compute_metrics(y_true, y_pred_array, y_pred_proba)
    print_metrics(metrics)
    save_results(metrics, model_name=args.model.upper())


if __name__ == "__main__":
    main()
