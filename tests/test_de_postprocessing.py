import numpy as np

from stellar_classification.DE_postprocessing import (
    apply_class_weights_to_probs,
    predict_with_class_weights,
)


def test_apply_class_weights_to_probs_multiplies_each_class_column():
    probs = np.array([
        [0.2, 0.5, 0.3],
        [0.6, 0.1, 0.3],
    ])
    weights = np.array([1.0, 2.0, 0.5])

    weighted_probs = apply_class_weights_to_probs(probs, weights)

    expected = np.array([
        [0.2, 1.0, 0.15],
        [0.6, 0.2, 0.15],
    ])
    np.testing.assert_allclose(weighted_probs, expected)


def test_predict_with_class_weights_returns_argmax_after_weighting():
    probs = np.array([
        [0.4, 0.3, 0.3],
        [0.2, 0.5, 0.3],
    ])
    weights = np.array([1.0, 0.5, 2.0])

    preds = predict_with_class_weights(probs, weights)

    np.testing.assert_array_equal(preds, np.array([2, 2]))
