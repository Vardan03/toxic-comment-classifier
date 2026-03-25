from src.data.load_data import load_train_data, load_test_data, load_test_labels
from src.data.preprocess import preprocess_train, preprocess_test
from train import train_model
from predict import predict, predict_proba
from src.utils.metrics import compute_metrics, print_metrics
from src.utils.logger import save_results


def main():
    # Loading all the data
    train = load_train_data()
    test = load_test_data()
    test_labels = load_test_labels()

    # Preprocessing
    train = preprocess_train(train)
    test, test_labels = preprocess_test(test, test_labels)

    # Train model (vectorization + training handled inside TFIDFModel)
    metrics = train_model(train)

    # Making predictions
    y_pred = predict(test["comment_text"])
    y_pred_proba = predict_proba(test["comment_text"])

    # Getting the actual labels
    y_true = test_labels.drop(columns=["id"]).values

    # Computing metrics
    metrics = compute_metrics(y_true, y_pred.values, y_pred_proba)
    print_metrics(metrics)
    save_results(metrics, model_name="baseline_tfidf")


if __name__ == "__main__":
    main()