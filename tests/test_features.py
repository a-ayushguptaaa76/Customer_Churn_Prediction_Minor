import pandas as pd
from src.features import add_features

def test_feature_engineering():
    sample = pd.DataFrame([{
        "customer_id": 1, "credit_score": 700, "country": "France",
        "gender": "Male", "age": 40, "tenure": 3, "balance": 80000,
        "products_number": 2, "credit_card": 1, "active_member": 1,
        "estimated_salary": 60000,
    }])
    out = add_features(sample)
    assert "customer_id" not in out
    for col in ["balance_per_product", "salary_balance_ratio", "age_group", "tenure_bucket", "high_balance"]:
        assert col in out
