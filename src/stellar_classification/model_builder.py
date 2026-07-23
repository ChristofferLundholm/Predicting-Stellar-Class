from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

from stellar_classification.preprocessing import make_preprocess_pipeline 


def make_lightgbm_model(params, random_state=1): 
    return LGBMClassifier(
        objective="multiclass",
        num_class=3,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
        **params,
    )


def make_model_pipeline(model):
    return Pipeline(
        steps=[
            ("preprocess", make_preprocess_pipeline()),
            ("model", model),
        ]
    )