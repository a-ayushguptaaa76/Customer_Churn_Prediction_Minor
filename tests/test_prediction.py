import pytest
from src.predict import predict_customer

def test_missing_artifact_is_explained(tmp_path):
    payload = {
        "credit_score": 700, "country": "France", "gender": "Male",
        "age": 40, "tenure": 3, "balance": 80000, "products_number": 2,
        "credit_card": 1, "active_member": 1, "estimated_salary": 60000,
    }
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        predict_customer(payload, model_path=tmp_path / "missing.joblib")
