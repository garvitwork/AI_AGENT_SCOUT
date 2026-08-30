"""
MLflow tracking setup pointed at DagsHub.
Get these 3 values from your DagsHub repo -> Remote -> Experiments tab:
  DAGSHUB_REPO_OWNER, DAGSHUB_REPO_NAME, DAGSHUB_TOKEN (DagsHub -> Settings -> Tokens)
Add them to .env.
"""
import os
import mlflow
from dotenv import load_dotenv

load_dotenv()

DAGSHUB_REPO_OWNER = os.getenv("DAGSHUB_REPO_OWNER")
DAGSHUB_REPO_NAME = os.getenv("DAGSHUB_REPO_NAME")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

TRACKING_URI = f"https://dagshub.com/{DAGSHUB_REPO_OWNER}/{DAGSHUB_REPO_NAME}.mlflow"

os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_REPO_OWNER or ""
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN or ""

def init_mlflow(experiment_name: str):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(experiment_name)
