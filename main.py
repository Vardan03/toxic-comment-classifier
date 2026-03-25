from src.data.load_data import load_train_data, load_test_data, load_test_labels, load_preprocess_train_data, load_preprocess_test_data, load_preprocess_test_labels
from src.data.preprocess import preprocess_and_save, preprocess_train, remove_unlabeled_rows
from src.models.train import train_model
from src.features.tfidf import fit_tfidf
from src.models.predict import predict, predict_proba
from src.utils.metrics import compute_metrics, print_metrics
from src.utils.logger import save_results

def main():
    train = load_train_data()
    preprocess_and_save()
    train = preprocess_train(train)
    
    vectors, vectorizer = fit_tfidf(train['comment_text'])
    train_model(vectors, train, vectorizer)

    test = load_preprocess_test_data()
    test_labels = load_preprocess_test_labels() 
    test_labels, test = remove_unlabeled_rows(test_labels, test)

    y_pred = predict(test["comment_text"])
    y_pred_proba = predict_proba(test["comment_text"])

    metrics = compute_metrics(test_labels.drop(columns=['id']).values, y_pred.values, y_pred_proba)
    print_metrics(metrics)
    save_results(metrics, model_name="trained_baseline_model")

if __name__ == "__main__":
    main()