from pathlib import Path

RANDOM_STATE = 1

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
SUBMISSION_DIR = PROJECT_ROOT / "submissions"

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"

MODEL_PATH = ARTIFACT_DIR / "lightgbm_pipeline.joblib"
LABEL_ENCODER_PATH = ARTIFACT_DIR / "label_encoder.joblib"
CLASS_WEIGHTS_PATH = ARTIFACT_DIR / "lightgbm_de_weights.npy"