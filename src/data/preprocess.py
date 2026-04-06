import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

def remove_ip_addresses(text):
    """Remove IP addresses like 192.168.0.1"""
    return re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '', text)

def remove_urls(text):
    """Remove URLs"""
    return re.sub(r'http\S+|www\S+', '', text)

def remove_html(text):
    """Remove HTML tags"""
    return re.sub(r'<.*?>', '', text)

def remove_wikipedia_artifacts(text):
    """Remove Wikipedia-specific noise"""
    return re.sub(r'\b(wikipedia|talk|user:).*$', '', text, flags=re.IGNORECASE)

def normalize_profanity(text):
    """Original word reconstruction"""
    patterns = {
        r'f[\W_]*u[\W_]*c[\W_]*k+': 'fuck',
        r's[\W_]*h[\W_]*i[\W_]*t+': 'shit',
        r'b[\W_]*i[\W_]*t[\W_]*c[\W_]*h+': 'bitch',
        r'a[\W_]*s[\W_]*s+': 'ass',
        r'd[\W_]*i[\W_]*c[\W_]*k+': 'dick',
        r'p[\W_]*u[\W_]*s[\W_]*s[\W_]*y+': 'pussy',
        r'c[\W_]*u[\W_]*n[\W_]*t+': 'cunt',
        r'n[\W_]*i[\W_]*g[\W_]*g[\W_]*e[\W_]*r+': 'nigger',
        r'f[\W_]*a[\W_]*g+': 'fag',
        r's[\W_]*l[\W_]*u[\W_]*t+': 'slut',
        r'w[\W_]*h[\W_]*o[\W_]*r[\W_]*e+': 'whore',
        r'b[\W_]*a[\W_]*s[\W_]*t[\W_]*a[\W_]*r[\W_]*d+': 'bastard'
    }
    for pattern, replacement in patterns.items():
        text = re.sub(pattern, replacement, text)

    return text


def normalize_repeated_chars(text):
    """Reduce repeated characters"""
    return re.sub(r'(.)\1{2,}', r'\1\1', text)


def remove_non_ascii(text):
    """Remove emojis and non-ASCII characters"""
    return text.encode('ascii', 'ignore').decode()


def remove_numbers(text):
    """Remove numbers"""
    return re.sub(r'\d+', '', text)


def clean_punctuation(text):
    """Remove punctuation except ? and !"""
    return re.sub(r'[^\w\s!?]', '', text)


def remove_extra_spaces(text):
    """Remove extra whitespace"""
    return re.sub(r'\s+', ' ', text).strip()


def remove_stopwords(text):
    """Remove common English stopwords"""
    return " ".join([w for w in text.split() if w not in ENGLISH_STOP_WORDS])


def lemmatize_text(text):
    """Convert words to classic form"""
    return " ".join([lemmatizer.lemmatize(w) for w in text.split()])

def clean_text(text):
    """
    Full text cleaning pipeline:
    - lowercasing
    - remove noise (URLs, HTML, IPs, Wikipedia artifacts)
    - normalize modification and repeated characters
    - remove non-ASCII and numbers
    - clean punctuation
    - normalize spacing
    """
    text = text.lower()
    text = remove_ip_addresses(text)
    text = remove_urls(text)
    text = remove_html(text)
    text = remove_wikipedia_artifacts(text)
    text = normalize_profanity(text)
    text = normalize_repeated_chars(text)
    text = remove_non_ascii(text)
    text = remove_numbers(text)
    text = clean_punctuation(text)
    text = remove_extra_spaces(text)
    return text

def add_text_features(df, text_column='comment_text'):
    """
    Add useful numerical features:
    - length of comment
    - word count
    - punctuation signals (!, ?)
    - uppercase intensity (important for toxicity)
    """
    df['comment_length'] = df[text_column].apply(len)
    df['word_count'] = df[text_column].apply(lambda x: len(x.split()))
    df['num_exclamation'] = df[text_column].apply(lambda x: x.count('!'))
    df['num_question'] = df[text_column].apply(lambda x: x.count('?'))
    df['num_uppercase'] = df[text_column].apply(lambda x: sum(1 for c in x if c.isupper()))
    return df

def preprocess_train(df, text_column='comment_text'):
    """
    Preprocess training data:
    - extract features BEFORE cleaning (to preserve uppercase)
    - clean text
    """
    df = add_text_features(df)
    df[text_column] = df[text_column].astype(str).apply(clean_text)
    return df


def remove_unlabeled_rows(labels: pd.DataFrame, test: pd.DataFrame):
    """Remove rows with label = -1"""
    LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    mask = (labels[LABELS] == -1).any(axis=1)
    return labels[~mask].reset_index(drop=True), test[~mask].reset_index(drop=True)


def preprocess_test(test, labels, text_column='comment_text'):
    """
    Preprocess test data:
    - remove unlabeled rows
    - extract features
    - clean text
    """
    labels_cleaned, test_cleaned = remove_unlabeled_rows(labels, test)
    test_cleaned = add_text_features(test_cleaned)
    test_cleaned[text_column] = test_cleaned[text_column].astype(str).apply(clean_text)
    return test_cleaned, labels_cleaned