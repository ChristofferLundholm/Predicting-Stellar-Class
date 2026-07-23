import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from stellar_classification.config import (
    ARTIFACT_DIR,
    CLASS_WEIGHTS_PATH,
    LABEL_ENCODER_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    TRAIN_PATH,
)
from stellar_classification.model_builder import make_lightgbm_model, make_model_pipeline


LIGHTGBM_BEST_PARAMS = {
    'n_estimators': 825,
    'learning_rate': 0.039346843032135655,
    'num_leaves': 106,
    'max_depth': 12,
    'min_child_samples': 128,
    'subsample': 0.7288467414126948,
    'colsample_bytree': 0.9046272936766726,
    'reg_alpha': 1.1191832145313731,
    'reg_lambda': 1.5304694288356494e-08
    }

LIGHTGBM_DE_WEIGHTS = np.array([
    0.50850416, 1.49608407, 1.99668201
])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the final LightGBM stellar-classification pipeline."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=TRAIN_PATH,
        help="Path to the training CSV.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=MODEL_PATH,
        help="Path where the fitted pipeline will be saved.",
    )
    parser.add_argument(
        "--label-encoder-output",
        type=Path,
        default=LABEL_ENCODER_PATH,
        help="Path where the fitted label encoder will be saved.",
    )
    parser.add_argument(
        "--class-weights-output",
        type=Path,
        default=CLASS_WEIGHTS_PATH,
        help="Path where the differential-evolution class weights will be saved.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.label_encoder_output.parent.mkdir(parents=True, exist_ok=True)
    args.class_weights_output.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(exist_ok=True)

    train_df = pd.read_csv(args.train_path)
    train_df = train_df.drop(columns="id")

    y = train_df.pop("class")
    X = train_df

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    model = make_lightgbm_model(
        params=LIGHTGBM_BEST_PARAMS,
        random_state=RANDOM_STATE,
    )
    pipeline = make_model_pipeline(model)

    pipeline.fit(X, y_encoded)

    joblib.dump(pipeline, args.model_output)
    joblib.dump(label_encoder, args.label_encoder_output)
    np.save(args.class_weights_output, LIGHTGBM_DE_WEIGHTS)

    print(f"Saved model to {args.model_output}")
    print(f"Saved label encoder to {args.label_encoder_output}")
    print(f"Saved class weights to {args.class_weights_output}")


if __name__ == "__main__":
    main()
