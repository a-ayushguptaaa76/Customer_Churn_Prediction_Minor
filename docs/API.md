# API

Run after training:
uvicorn app.api:app --reload

OpenAPI documentation is available at /docs.

GET /health returns service status.

POST /predict accepts customer attributes and an optional classification threshold.

Example JSON:
{
  "credit_score": 650,
  "country": "France",
  "gender": "Male",
  "age": 40,
  "tenure": 3,
  "balance": 50000,
  "products_number": 2,
  "credit_card": 1,
  "active_member": 1,
  "estimated_salary": 60000,
  "threshold": 0.50
}

The response contains churn_prediction, churn_probability, risk_band, recommended_action, and threshold.
