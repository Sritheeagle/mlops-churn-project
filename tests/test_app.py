from fastapi.testclient import TestClient

from src.app import app


client = TestClient(app)


def test_health_reports_trained_model() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_returns_churn_prediction() -> None:
    response = client.post(
        "/predict",
        json={
            "tenure_months": 6,
            "monthly_charges": 95,
            "total_charges": 570,
            "support_tickets": 4,
            "contract_type_code": 0,
            "payment_delay_days": 12,
        },
    )

    assert response.status_code == 200
    result = response.json()
    assert result["churn_prediction"] in (0, 1)
    assert 0 <= result["churn_probability"] <= 1
    assert result["risk_tier"] in ("LOW", "MEDIUM", "HIGH")


def test_predict_rejects_out_of_range_features() -> None:
    response = client.post(
        "/predict",
        json={
            "tenure_months": 121,
            "monthly_charges": 95,
            "total_charges": 570,
            "support_tickets": 4,
            "contract_type_code": 0,
            "payment_delay_days": 12,
        },
    )

    assert response.status_code == 422
