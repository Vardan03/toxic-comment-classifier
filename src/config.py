from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent  # project root

# Data paths
RAW_TRAIN_PATH              = ROOT_DIR / "data" / "raw" / "train.csv"
RAW_TEST_PATH               = ROOT_DIR / "data" / "raw" / "test.csv"
RAW_TEST_LABELS_PATH        = ROOT_DIR / "data" / "raw" / "test_labels.csv"

PROCESSED_TRAIN_PATH        = ROOT_DIR / "data" / "processed" / "cleaned_train.csv"
PROCESSED_TEST_PATH         = ROOT_DIR / "data" / "processed" / "cleaned_test.csv"
PROCESSED_TEST_LABELS_PATH  = ROOT_DIR / "data" / "processed" / "cleaned_test_labels.csv"

# Model & reports
MODELS_PATH  = ROOT_DIR / "saved_models"

# per model paths
TFIDF_PATH   = MODELS_PATH / "tfidf" / "tfidf_model.pkl"
RNN_PATH     = MODELS_PATH / "rnn"
LSTM_PATH    = MODELS_PATH / "lstm"
BERT_PATH    = MODELS_PATH / "bert"
GPT_PATH     = MODELS_PATH / "gpt"

REPORTS_PATH = ROOT_DIR / "reports" / "result.txt"

# NLP / Features
MAX_FEATURES = 20000
NGRAM_RANGE  = (1, 2)
THRESHOLD = 0.3

RANDOM_STATE = 42
TEST_SIZE    = 0.2

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
