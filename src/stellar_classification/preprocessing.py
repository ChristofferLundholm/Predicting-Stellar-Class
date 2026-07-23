
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


class AlphaSinCosTransformer(BaseEstimator, TransformerMixin):
    """Replace alpha degrees with sine/cosine features."""

    def __init__(self, alpha_col="alpha", drop_original=True):
        self.alpha_col = alpha_col
        self.drop_original = drop_original

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X = X.copy()
        alpha_rad = np.deg2rad(X[self.alpha_col])
        X["alpha_sin"] = np.sin(alpha_rad)
        X["alpha_cos"] = np.cos(alpha_rad)

        if self.drop_original:
            X = X.drop(columns=[self.alpha_col])

        return X


class DropColumnsTransformer(BaseEstimator, TransformerMixin):
    """Drop columns if they are present, so train/test transforms stay aligned."""

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        return X.drop(columns=self.columns, errors="ignore")

def make_preprocess_pipeline():

    return Pipeline(
        steps=[
            ("alpha_sin_cos", AlphaSinCosTransformer()),
            ("drop_synthetic_categoricals", DropColumnsTransformer(["spectral_type", "galaxy_population"])),
        ]
    )