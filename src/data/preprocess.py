import re
import pandas as pd
from src.config import PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH, PROCESSED_TEST_LABELS_PATH
from src.data.load_data import load_train_data, load_test_data, load_test_labels

def remove_ip_addresses(text):
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    return re.sub(ip_pattern, '', text)     # Remove special characters

def clean_text(text):
    text = text.lower()
    text = remove_ip_addresses(text)
    return text

def preprocess_train(df, text_column='comment_text'):
    df[text_column] = df[text_column].astype(str).apply(clean_text)
    return df

def preprocess_and_save():
    train = load_train_data()
    train_cleaned = preprocess_train(train)
    train_cleaned.to_csv(PROCESSED_TRAIN_PATH, index=False)
    test_labels = load_test_labels()
    test = load_test_data()
    test_labels_cleaned, test_cleaned = remove_unlabeled_rows(test_labels, test)
    test_labels_cleaned.to_csv(PROCESSED_TEST_LABELS_PATH, index=False)
    test_cleaned.to_csv(PROCESSED_TEST_PATH, index=False)
    print("Preprocessing done! Saved to", PROCESSED_TRAIN_PATH)

def remove_unlabeled_rows(labels: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    mask = (labels[LABELS] == -1).any(axis=1)
    return labels[~mask].reset_index(drop=True), test[~mask].reset_index(drop=True)