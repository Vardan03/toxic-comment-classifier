from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from src.config import MODEL_PATH, TFIDF_PATH, RANDOM_STATE, TEST_SIZE, MAX_FEATURES, NGRAM_RANGE

def fit_tfidf(corpus):
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
    X = vectorizer.fit_transform(corpus)
    with open(TFIDF_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
    return X, vectorizer

def transform_tfidf(corpus, vectorizer):
    return vectorizer.transform(corpus)