import re
import pandas as pd
from src.config import PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH

def remove_ip_addresses(text):
    ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
    return re.sub(ip_pattern, '', text)     # Remove special characters

def clean_text(text):
    text = text.lower()
    text = remove_ip_addresses(text)
    return text

def preprocess_df(df, text_column='comment_text'):
    df[text_column] = df[text_column].astype(str).apply(clean_text)
    return df

def preprocess_and_save():
    train = pd.read_csv("data/raw/train.csv")
    train_cleaned = preprocess_df(train)
    train_cleaned.to_csv(PROCESSED_TRAIN_PATH, index=False)
    print("Preprocessing done! Saved to", PROCESSED_TRAIN_PATH)