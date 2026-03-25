from src.data.load_data import load_train_data, load_test_data, load_test_labels
from src.data.preprocess import preprocess_and_save, preprocess_df, remove_unlabeled_rows
from src.models.train import train_model
from src.features.tfidf import fit_tfidf
from src.models.predict import evaluate

def main():
    df = load_train_data()
    preprocess_and_save()
    df = preprocess_df(df)
    
    X, vectorizer = fit_tfidf(df['comment_text'])
    train_model(X, df, vectorizer)

    test = load_test_data()
    test_labels = load_test_labels()
    test_labels, test = remove_unlabeled_rows(test_labels, test)

    evaluate(test["comment_text"], test_labels.drop(columns=['id']).values)

if __name__ == "__main__":
    main()