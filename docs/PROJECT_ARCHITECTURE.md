# Project Architecture

Data flow:
CSV -> validation -> feature engineering -> preprocessing -> cross-validation -> candidate models -> held-out evaluation -> serialized pipeline -> FastAPI prediction service

The feature function is shared by training and inference to reduce training-serving skew.

Numeric columns are median-imputed and standardized; categorical columns are imputed and one-hot encoded.

Training compares Logistic Regression, Random Forest, and Gradient Boosting with stratified 5-fold ROC-AUC.

Evaluation reports accuracy, precision, recall, F1, and ROC-AUC because churn is an imbalanced target.

FastAPI exposes /health and /predict.

Future MLOps extensions can include experiment tracking, calibration, drift monitoring, a model registry, scheduled retraining, and cloud deployment.
