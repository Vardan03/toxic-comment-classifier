import pandas as pd
from src.config import RAW_TRAIN_PATH, RAW_TEST_PATH, TEST_LABELS_PATH, PROCESSED_TRAIN_PATH, PROCESSED_TEST_PATH, PROCESSED_TEST_LABELS_PATH

def load_train_data():
    return pd.read_csv(RAW_TRAIN_PATH)

def load_test_data():
    return pd.read_csv(RAW_TEST_PATH)

def load_test_labels():
    return pd.read_csv(TEST_LABELS_PATH)
    

def load_preprocess_train_data():
    return pd.read_csv(PROCESSED_TRAIN_PATH)

def load_preprocess_test_data():
    return pd.read_csv(PROCESSED_TEST_PATH)

def load_preprocess_test_labels():
    return pd.read_csv(PROCESSED_TEST_LABELS_PATH)