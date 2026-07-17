# Stellar Classification

This project develops a reproducible machine-learning pipeline for classifying astronomical objects from tabular observational data. The dataset comes from the Kaggle Playground Series S6E6 competition and contains three target classes: galaxies, quasars, and stars.

The project covers exploratory data analysis, feature engineering, model comparison, out-of-fold evaluation, hyperparameter optimisation, ensembling, and submission generation. It is also being extended with SQL-based data handling and broader MLOps practices.

Kaggle Competition link: https://www.kaggle.com/competitions/playground-series-s6e6/overview. 

## Current Results
- Built an exploratory analysis and baseline modelling notebook using LightGBM.
- Analysed class imbalance, feature distributions, mutual information, and grouped feature ablations.
- Implemented circular encoding for angular variables and evaluated the contribution of positional, photometric, and redshift features.
- Optimised class-probability adjustments using differential evolution within a leakage-resistant cross-validation procedure.
- Achieved a Kaggle private score of 0.9635.
- Developed a second modelling pipeline comparing LightGBM, CatBoost, RealMLP, and Random Forest using reusable scikit-learn pipelines and out-of-fold predictions.

## Project Structure
stellar-classification/
├── notebooks/
│   ├── 01_eda_and_baseline_model.ipynb
│   └── 02_model_selection_stacking_and_optimization.ipynb
├── src/
├── README.md
└── requirements.txt

## Methods and Tools

Python, pandas, NumPy, scikit-learn, LightGBM, CatBoost, RealMLP, Random Forest, Optuna, SciPy, and Kaggle.

## Evaluation

Models are evaluated using cross-validation and out-of-fold predictions. Optimisation steps that depend on model probabilities are performed on held-out predictions rather than in-sample outputs to reduce the risk of optimistic evaluation.

## Planned Work
Complete model selection and hyperparameter optimisation.
Evaluate stacking and blending approaches.
Refactor repeated notebook logic into reusable Python modules.


