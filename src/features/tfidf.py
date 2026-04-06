from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from src.config import MAX_FEATURES, NGRAM_RANGE

def fit_tfidf(corpus):
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES, ngram_range=NGRAM_RANGE)
    X = vectorizer.fit_transform(corpus)
    return X, vectorizer

def transform_tfidf(corpus, vectorizer):
    return vectorizer.transform(corpus)