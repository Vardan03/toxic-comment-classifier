from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from src.config import MAX_FEATURES, NGRAM_RANGE, MODEL_PATH

def fit_tfidf(corpus):
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
    X = vectorizer.fit_transform(corpus)
    with open(MODEL_PATH.with_name(MODEL_PATH.stem + '_tfidf.pkl'), 'wb') as f:
        pickle.dump(vectorizer, f)
    return X, vectorizer

def transform_tfidf(corpus, vectorizer):
    return vectorizer.transform(corpus)