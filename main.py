from src.data.load_data import load_train_data, load_test_data, load_test_labels, load_preprocess_train_data, load_preprocess_test_data, load_preprocess_test_labels
from src.data.preprocess import preprocess_train, preprocess_test
from train import train_model, train_rnn_model
from predict import predict, predict_proba, predict_rnn, predict_rnn_proba
from src.utils.metrics import compute_metrics, print_metrics
from src.utils.logger import save_results
from src.config import LABEL_COLS
import numpy as np

def main():
    # Loading all the data
    # train = load_train_data()
    train = load_preprocess_train_data()
    test = load_test_data()
    test_labels = load_test_labels()

    # Preprocessing
    # train = preprocess_train(train)
    test, test_labels = preprocess_test(test, test_labels)

    # Train model (vectorization + training handled inside TFIDFModel)
    # metrics = train_model(train)

    # Train RNN model
    # train_y = train[LABEL_COLS].to_numpy().astype(np.float32)
    # model_rnn = train_rnn_model(train['comment_text'], train_y)

    # Making predictions
    # y_pred = predict(test["comment_text"])
    # y_pred_proba = predict_proba(test["comment_text"])
    y_pred = predict_rnn(test["comment_text"])
    print(y_pred)
    y_pred_proba = predict_rnn_proba(test["comment_text"])

    # Getting the actual labels
    y_true = test_labels.drop(columns=["id"]).values

    # Computing metrics
    metrics = compute_metrics(y_true, y_pred, y_pred_proba)
    print_metrics(metrics)
    save_results(metrics, model_name="RNNModel")



if __name__ == "__main__":
    main()