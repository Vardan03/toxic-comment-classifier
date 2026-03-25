from src.data.load_data import load_train_data, load_test_data, load_test_labels, load_preprocess_train_data, load_preprocess_test_data, load_preprocess_test_labels
from src.data.preprocess import preprocess_and_save, preprocess_df, remove_unlabeled_rows
from src.models.train import train_model
from src.features.tfidf import fit_tfidf
from src.models.predict import predict, predict_proba
from src.utils.metrics import compute_metrics, print_metrics
from src.utils.logger import save_results

def main():
    df = load_preprocess_train_data()
    # preprocess_and_save()
    # df = preprocess_df(df)
    
    X, vectorizer = fit_tfidf(df['comment_text'])
    train_model(X, df, vectorizer)

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