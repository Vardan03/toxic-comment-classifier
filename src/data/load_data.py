import pandas as pd
from src.config import RAW_TRAIN_PATH, RAW_TEST_PATH

def load_train_data():
    return pd.read_csv(RAW_TRAIN_PATH)

def load_test_data():
    return pd.read_csv(RAW_TEST_PATH)

if __name__ == "__main__":
    train = load_train_data()
    test = load_test_data()
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)