# Stellar Classification

This project develops a tabular machine-learning workflow for classifying astronomical objects as galaxies, quasars, or stars. The dataset comes from the Kaggle Playground Series S6E6 competition.

The project covers exploratory data analysis, feature engineering, feature-group ablation, model comparison, out-of-fold validation, hyperparameter optimization, probability post-processing, stacking experiments, and Kaggle submission generation.

Kaggle competition: https://www.kaggle.com/competitions/playground-series-s6e6/over

## Reproducibility

Place the Kaggle competition files in `data/train.csv` and `data/test.csv`, then install the project in editable mode:

```powershell
pip install -e ".[dev]"
```

Train the final LightGBM pipeline and save the local artifacts:

```powershell
python -m stellar_classification.train
```

Generate a Kaggle submission from the saved model, label encoder, and probability weights:

```powershell
python -m stellar_classification.predict
```

Optional paths can be overridden from the CLI:

```powershell
python -m stellar_classification.train --train-path data/train.csv --model-output artifacts/lightgbm_pipeline.joblib
python -m stellar_classification.predict --input-path data/test.csv --output-path submissions/submission.csv
```

Run the tests:

```powershell
pytest
```

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
data/                         Kaggle input files, ignored by git
notebooks/                    EDA, baseline modeling, HPO, and stacking experiments
src/stellar_classification/   Reusable preprocessing, training, prediction, and post-processing code
tests/                        Unit tests for reusable project code
artifacts/                    Trained model artifacts, ignored by git
submissions/                  Generated Kaggle submissions, ignored by git
pyproject.toml                Project metadata, dependencies, and pytest configuration
README.md                     Project overview and reproduction notes
```

## Methods and Tools

Python, pandas, NumPy, scikit-learn, LightGBM, CatBoost, RealMLP, Optuna, SciPy, Matplotlib, Seaborn, and Kaggle.

## Evaluation

Models are evaluated with stratified cross-validation and out-of-fold predictions using balanced accuracy. Probability-based post-processing is evaluated on held-out OOF predictions to reduce optimistic estimates.

## Final Model Choice
Although stacking was tested, it did not outperform the simpler tuned LightGBM model with differential-evolution probability weighting. The final submission therefore prioritizes validation performance, simplicity, and robustness over additional ensemble complexity.

