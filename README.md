# Customer Churn Prediction

An end-to-end machine-learning portfolio project for predicting customer churn, explaining risk drivers, and exposing a reusable prediction workflow.

> **Status:** Portfolio-ready foundation under active development.

## Why this project

Customer churn is a classification problem where missing a likely-to-leave customer can matter more than maximizing raw accuracy. This project therefore focuses on reproducible preprocessing, model comparison, probability-based predictions, threshold-aware evaluation, and explainable insights.

## Current baseline

The source dataset contains **10,000 customer records**, 11 input variables, and a binary `churn` target. The existing experiment recorded a **5-fold cross-validated ROC-AUC of 0.8628** for Gradient Boosting and a held-out **ROC-AUC of 0.8692** with **86.80% accuracy**.

These figures are preserved as **baseline results from the original experiment**. The upgraded repository separates baseline reporting from newly reproducible training code so that future results can be traced to an explicit run.

## Planned production-style workflow

```text
Raw customer data
      |
      v
Validation + cleaning
      |
      v
Feature engineering
      |
      v
Train / validation split
      |
      v
Preprocessing pipeline
      |
      +-------------------------------+
      |               |               |
      v               v               v
Logistic Regression  Random Forest  Gradient Boosting
      |               |               |
      +---------------+---------------+
                      |
                      v
               Model evaluation
                      |
            +---------+---------+
            |                   |
            v                   v
       Risk scoring       Explainability
            |                   |
            +---------+---------+
                      |
                      v
                API / Dashboard
```

## Repository structure

```text
Customer_Churn_Prediction_Minor/
├── app/
│   └── api.py
├── data/
│   └── README.md
├── docs/
│   ├── MODEL_CARD.md
│   ├── PROJECT_ARCHITECTURE.md
│   └── API.md
├── models/
│   └── README.md
├── notebooks/
│   └── customer_churn_analysis.ipynb
├── src/
│   ├── __init__.py
│   ├── features.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/
│   ├── test_features.py
│   ├── test_prediction.py
│   └── test_api.py
├── .github/workflows/ci.yml
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── LICENSE
└── README.md
```

## Tech stack

**Python · Pandas · NumPy · Scikit-learn · XGBoost-ready workflow · FastAPI · Streamlit-ready architecture · Docker · GitHub Actions**

## Key engineering decisions

- The customer identifier is excluded from model features.
- Categorical variables are encoded inside a pipeline to reduce training/inference mismatch.
- Numerical preprocessing is fitted only on training data.
- Stratified splitting is used because churn is an imbalanced target.
- ROC-AUC, precision, recall, and F1 are reported together rather than relying on accuracy alone.
- Threshold tuning is treated as a business decision, not a model-quality shortcut.
- Probability outputs are intended for prioritization, not as a guarantee that a customer will churn.

## Baseline experiment findings

The baseline experiment compared Logistic Regression, Random Forest, Gradient Boosting, AdaBoost, and SVC. Recorded mean 5-fold ROC-AUC values were:

| Model | CV ROC-AUC |
|---|---:|
| Logistic Regression | 0.7877 |
| Random Forest | 0.8486 |
| Gradient Boosting | 0.8628 |
| AdaBoost | 0.8462 |
| SVC | 0.8351 |

The baseline Gradient Boosting test evaluation was:

| Metric | Value |
|---|---:|
| Accuracy | 0.8680 |
| Precision | 0.7804 |
| Recall | 0.4889 |
| F1 | 0.6012 |
| ROC-AUC | 0.8692 |

Because recall is materially lower than accuracy, the upgraded project will include threshold analysis rather than presenting the baseline as a complete retention solution.

## Example prediction concept

The final system is designed to accept customer attributes and return:

```json
{
  "churn_probability": 0.31,
  "risk_band": "medium",
  "recommended_action": "review retention offer"
}
```

The API layer is deliberately separated from training so the model can later be deployed independently.

## Dataset note

The initial dataset/schema was carried over from the earlier Customer Churn Prediction project used as the baseline. The model engineering, repository structure, documentation, and application layer in this repository are being rebuilt as a standalone portfolio project. Dataset provenance and permitted reuse should be reviewed before commercial use.

## Roadmap

- [x] Preserve and clarify the baseline experiment
- [x] Establish a standalone repository structure
- [ ] Reproducible training CLI
- [ ] Threshold tuning and calibration
- [ ] Explainability with feature attribution
- [ ] REST prediction API
- [ ] Interactive dashboard
- [ ] Containerized deployment
- [ ] CI quality gates
- [ ] Cloud deployment guide

## Author

**Ayush Kumar Gupta**

[GitHub](https://github.com/Blood79) · [LinkedIn](https://linkedin.com/in/ayush-kumar-gupta-43314b238)
