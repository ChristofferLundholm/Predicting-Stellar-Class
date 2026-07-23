import numpy as np

#Use already learnt DE weights.

def apply_class_weights_to_probs(probs, weights):
    return np.asarray(probs, dtype=float) * np.asarray(weights, dtype=float)


def predict_with_class_weights(probs, weights):
    weighted_probs = apply_class_weights_to_probs(probs, weights)
    return weighted_probs.argmax(axis=1)