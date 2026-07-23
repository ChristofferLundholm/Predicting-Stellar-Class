# Stellar Classification

This project develops a tabular machine-learning workflow for classifying astronomical objects as galaxies, quasars, or stars. The dataset comes from the Kaggle Playground Series S6E6 competition.

The project covers exploratory data analysis, feature engineering, feature-group ablation, model comparison, out-of-fold validation, hyperparameter optimization, probability post-processing, stacking experiments, and Kaggle submission generation.

Kaggle competition: https://www.kaggle.com/competitions/playground-series-s6e6/overview

## Reproducibility Commands
pip install -e ".[dev]"
python -m stellar_classification.train
python -m stellar_classification.predict --input data/test.csv --output submissions/submission.csv
pytest

## Results

The final selected model is a tuned LightGBM classifier with differential-evolution class probability weighting.

- Public leaderboard score: 0.96741
- Private leaderboard score: 0.96694
- Approximate private rank: 714 / 2817

A simpler LightGBM baseline with probability weighting achieved:

- Public leaderboard score: 0.9643
- Private leaderboard score: 0.9635

## What This Project Demonstrates

- Exploratory analysis of class imbalance, feature distributions, and class-level feature behavior.
- Circular encoding of the angular `alpha` coordinate using sine/cosine features.
- Feature-group ablation to compare redshift, photometric bands, positional features, synthetic categorical features, and photometric differences.
- Reusable scikit-learn preprocessing and model pipelines.
- Stratified out-of-fold validation for fair model comparison.
- Model comparison across LightGBM, CatBoost, RandomForest, and RealMLP.
- Hyperparameter optimization with Optuna.
- Probability weighting with SciPy differential evolution to improve balanced accuracy.
- Stacking experiments using OOF probabilities, with final model selection based on validation performance and simplicity.

## Project Structure

```
├── data/
├── notebooks/
│   ├── 01_eda_and_ baseline_model.ipynb
│   └── 02_model_selection_stacking_and_optimization.ipynb
├── README.md
└── requirements.txt
```

## Methods and Tools

Python, pandas, NumPy, scikit-learn, LightGBM, CatBoost, RealMLP, Optuna, SciPy, Matplotlib, Seaborn, and Kaggle.

## Evaluation

Models are evaluated with stratified cross-validation and out-of-fold predictions using balanced accuracy. Probability-based post-processing is evaluated on held-out OOF predictions to reduce optimistic estimates.

## Final Model Choice
Although stacking was tested, it did not outperform the simpler tuned LightGBM model with differential-evolution probability weighting. The final submission therefore prioritizes validation performance, simplicity, and robustness over additional ensemble complexity.


