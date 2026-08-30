import sys, os
import mlflow

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ml"))
from mlflow_config import init_mlflow, TRACKING_URI

RISK_MODEL_NAME = "scout-risk-classifier"
DELAY_MODEL_NAME = "scout-delay-forecaster"


def _load_latest(model_name: str):
    mlflow.set_tracking_uri(TRACKING_URI)
    client = mlflow.MlflowClient()
    latest = client.get_latest_versions(model_name)[0]
    return mlflow.xgboost.load_model(f"models:/{model_name}/{latest.version}")


def load_models():
    init_mlflow("scout_inference")  # dummy call just to ensure tracking URI/auth is set
    risk_model = _load_latest(RISK_MODEL_NAME)
    delay_model = _load_latest(DELAY_MODEL_NAME)
    return risk_model, delay_model
