import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from stellar_classification.config import (
    CLASS_WEIGHTS_PATH,
    LABEL_ENCODER_PATH,
    MODEL_PATH,
    SUBMISSION_DIR,
    TEST_PATH,
)
from stellar_classification.DE_postprocessing import predict_with_class_weights


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a Kaggle submission from the trained stellar classifier."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=TEST_PATH,
        help="Path to the test CSV.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=SUBMISSION_DIR / "submission_lightgbm_hpo_de.csv",
        help="Path where the submission CSV will be saved.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help="Path to the fitted pipeline artifact.",
    )
    parser.add_argument(
        "--label-encoder-path",
        type=Path,
        default=LABEL_ENCODER_PATH,
        help="Path to the fitted label encoder artifact.",
    )
    parser.add_argument(
        "--class-weights-path",
        type=Path,
        default=CLASS_WEIGHTS_PATH,
        help="Path to the differential-evolution class weights artifact.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    test_df = pd.read_csv(args.input_path)
    test_ids = test_df.pop("id")

    pipeline = joblib.load(args.model_path)
    label_encoder = joblib.load(args.label_encoder_path)
    class_weights = np.load(args.class_weights_path)

    probs = pipeline.predict_proba(test_df)
    preds = predict_with_class_weights(probs, class_weights)
    pred_labels = label_encoder.inverse_transform(preds)

    submission = pd.DataFrame(
        {
            "id": test_ids,
            "class": pred_labels,
        }
    )

    submission.to_csv(args.output_path, index=False)

    print(f"Saved submission to {args.output_path}")


if __name__ == "__main__":
    main()
