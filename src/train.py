from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
EXPERIMENT_NAME = "customer-churn-production"
RANDOM_STATE = 42


def make_synthetic_churn_data(
    n_customers: int = 2_000,
) -> tuple[pd.DataFrame, pd.Series]:
    """Generate a reproducible, synthetic dataset for demonstrating the workflow."""
    rng = np.random.default_rng(RANDOM_STATE)
    tenure_months = rng.integers(1, 73, size=n_customers)
    monthly_charges = rng.normal(70, 20, size=n_customers).clip(20, 130)
    features = pd.DataFrame(
        {
            "tenure_months": tenure_months,
            "monthly_charges": monthly_charges,
            "total_charges": (
                rng.uniform(0.8, 1.2, size=n_customers)
                * tenure_months
                * monthly_charges
            ),
            "support_tickets": rng.poisson(1.5, size=n_customers),
            "contract_type_code": rng.choice(
                [0, 1, 2], size=n_customers, p=[0.5, 0.3, 0.2]
            ),
            "payment_delay_days": rng.integers(0, 31, size=n_customers),
        }
    )
    log_odds = (
        1.3
        - 0.045 * features["tenure_months"]
        + 0.018 * (features["monthly_charges"] - 70)
        + 0.35 * features["support_tickets"]
        + 0.12 * features["contract_type_code"]
        + 0.025 * features["payment_delay_days"]
    )
    churn_probability = 1 / (1 + np.exp(-log_odds))
    churned = pd.Series(
        rng.binomial(1, churn_probability), name="churn"
    )
    return features, churned


def main() -> None:
    features, target = make_synthetic_churn_data()
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1_000, random_state=RANDOM_STATE),
    )

    mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))

        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "dataset": "synthetic_churn",
                "n_customers": len(features),
                "test_size": 0.2,
                "random_state": RANDOM_STATE,
                "max_iter": 1_000,
                "features": ",".join(features.columns),
            }
        )
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, artifact_path="model")

        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)

    print(f"Training Complete | Accuracy: {accuracy:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
