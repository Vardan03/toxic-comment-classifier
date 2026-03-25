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
MODEL_PATH   = ROOT_DIR / "saved_models" / "baseline_model.pkl"
TFIDF_PATH = ROOT_DIR / "saved_models" / "baseline_model_tfidf.pkl"
REPORTS_PATH = ROOT_DIR / "reports" / "result.txt"

# NLP / Features
MAX_FEATURES = 20000
NGRAM_RANGE  = (1, 2)

RANDOM_STATE = 42
TEST_SIZE    = 0.2