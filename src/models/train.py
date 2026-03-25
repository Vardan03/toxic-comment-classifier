import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from src.data.load_data import load_train_data
from src.data.preprocess import preprocess_df
from src.features.tfidf import fit_tfidf
from src.config import MODEL_PATH, RANDOM_STATE, TEST_SIZE
from src.utils.logger import save_results

def train_model(X, df, vectorizer):
    y = df[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    model = OneVsRestClassifier(LogisticRegression(random_state=RANDOM_STATE, max_iter=1000))
    model.fit(X_train, y_train)

    # Save model
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump((model, vectorizer), f)
    print("Model trained and saved at", MODEL_PATH)

if __name__ == "__main__":
    train_model()