from pathlib import Path
import joblib
import pandas as pd
from .evaluate import apply_threshold
from .features import add_features

DEFAULT_MODEL_PATH = Path("models/churn_pipeline.joblib")

def load_model(model_path=DEFAULT_MODEL_PATH):
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {path}. "
            "Run 'python -m src.train --data data/customer_data.csv' first."
        )
    return joblib.load(path)

def predict_customer(payload: dict, *, threshold: float = 0.50, model_path=DEFAULT_MODEL_PATH) -> dict:
    model = load_model(model_path)
    frame = add_features(pd.DataFrame([payload]))
    probability = float(model.predict_proba(frame)[:, 1][0])
    prediction = int(apply_threshold([probability], threshold)[0])
    if probability >= 0.70:
        risk_band, action = "high", "prioritize retention outreach"
    elif probability >= 0.40:
        risk_band, action = "medium", "review for targeted retention"
    else:
        risk_band, action = "low", "monitor normally"
    return {
        "churn_prediction": prediction,
        "churn_probability": round(probability, 4),
        "risk_band": risk_band,
        "recommended_action": action,
        "threshold": threshold,
    }
