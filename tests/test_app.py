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

    metrics_response = client.get("/metrics")
    assert metrics_response.status_code == 200
    assert "churn_api_http_requests_total" in metrics_response.text
    assert "churn_api_http_request_duration_seconds" in metrics_response.text
    assert "churn_api_predictions_total" in metrics_response.text
    assert "churn_api_prediction_probability" in metrics_response.text
    assert (
        f'churn_api_predictions_total{{prediction="{result["churn_prediction"]}",'
        f'risk_tier="{result["risk_tier"]}"}}'
    ) in metrics_response.text


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
