from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from .evaluate import binary_metrics
from .features import add_features
from .preprocessing import build_preprocessor

def build_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, random_state=42,
            class_weight="balanced_subsample", n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=250, learning_rate=0.05, max_depth=3, random_state=42
        ),
    }

def train(data_path="data/customer_data.csv", model_dir="models"):
    df = pd.read_csv(data_path)
    if "churn" not in df.columns:
        raise ValueError("Dataset must contain a 'churn' target column.")
    X = add_features(df.drop(columns=["churn"]))
    y = df["churn"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []
    for name, classifier in build_models().items():
        pipe = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ])
        scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.50).astype(int)
        metrics = binary_metrics(y_test, pred, proba)
        metrics.update(model=name, cv_roc_auc_mean=float(scores.mean()), cv_roc_auc_std=float(scores.std()))
        results.append(metrics)
    results.sort(key=lambda r: r["cv_roc_auc_mean"], reverse=True)
    best_name = results[0]["model"]
    best = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", build_models()[best_name]),
    ])
    best.fit(X_train, y_train)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    artifact = Path(model_dir) / "churn_pipeline.joblib"
    joblib.dump(best, artifact)
    report = {
        "best_model": best_name,
        "dataset_rows": len(df),
        "churn_rate": float(y.mean()),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "results": results,
        "artifact": str(artifact),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/customer_data.csv")
    parser.add_argument("--model-dir", default="models")
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.model_dir), indent=2))
