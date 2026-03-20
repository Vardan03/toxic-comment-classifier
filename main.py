from src.data.load_data import load_train_data
from src.data.preprocess import preprocess_and_save, preprocess_df
from src.models.train import train_model

def main():
    df = load_train_data()
    preprocess_and_save()
    df = preprocess_df(df)
    
    X, vectorizer = fit_tfidf(df['comment_text'])
    train_model(X, df, vectorizer)

if __name__ == "__main__":
    main()