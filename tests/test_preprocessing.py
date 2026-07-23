import pandas as pd

from stellar_classification.preprocessing import make_preprocess_pipeline


def test_preprocessing_creates_alpha_sin_cos():
    X = pd.DataFrame({
        "alpha": [0.0, 90.0, 180.0],
        "delta": [1.0, 2.0, 3.0],
        "u": [1.0, 1.0, 1.0],
        "g": [1.0, 1.0, 1.0],
        "r": [1.0, 1.0, 1.0],
        "i": [1.0, 1.0, 1.0],
        "z": [1.0, 1.0, 1.0],
        "redshift": [0.1, 0.2, 0.3],
        "spectral_type": ["A", "B", "C"],
        "galaxy_population": ["x", "y", "z"],
    })

    pipeline = make_preprocess_pipeline()
    X_transformed = pipeline.fit_transform(X)

    assert "alpha_sin" in X_transformed.columns
    assert "alpha_cos" in X_transformed.columns


def test_preprocessing_drops_unused_columns():
    X = pd.DataFrame({
        "alpha": [0.0],
        "delta": [1.0],
        "u": [1.0],
        "g": [1.0],
        "r": [1.0],
        "i": [1.0],
        "z": [1.0],
        "redshift": [0.1],
        "spectral_type": ["A"],
        "galaxy_population": ["x"],
    })

    pipeline = make_preprocess_pipeline()
    X_transformed = pipeline.fit_transform(X)

    assert "alpha" not in X_transformed.columns
    assert "spectral_type" not in X_transformed.columns
    assert "galaxy_population" not in X_transformed.columns
