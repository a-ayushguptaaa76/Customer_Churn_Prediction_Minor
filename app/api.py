from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.predict import predict_customer

app = FastAPI(
    title="Customer Churn Prediction API",
    version="1.0.0",
    description="Probability-based customer churn scoring API.",
)

class Customer(BaseModel):
    credit_score: int = Field(ge=300, le=900)
    country: str
    gender: str
    age: int = Field(ge=18, le=100)
    tenure: int = Field(ge=0)
    balance: float = Field(ge=0)
    products_number: int = Field(ge=1)
    credit_card: int = Field(ge=0, le=1)
    active_member: int = Field(ge=0, le=1)
    estimated_salary: float = Field(ge=0)
    threshold: float = Field(default=0.50, gt=0, lt=1)

@app.get("/health")
def health():
    return {"status": "ok", "model_path": str(Path("models/churn_pipeline.joblib"))}

@app.post("/predict")
def predict(customer: Customer):
    data = customer.model_dump()
    threshold = data.pop("threshold")
    try:
        return predict_customer(data, threshold=threshold)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
