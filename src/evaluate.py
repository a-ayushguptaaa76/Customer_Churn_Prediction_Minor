from __future__ import annotations
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

def binary_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    y_true, y_pred, y_proba = map(np.asarray, (y_true, y_pred, y_proba))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
    }

def apply_threshold(y_proba, threshold: float = 0.50) -> np.ndarray:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1")
    return (np.asarray(y_proba) >= threshold).astype(int)
