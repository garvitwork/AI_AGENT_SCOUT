"""
Trains the shipment disruption-risk classifier (is_delayed) and logs
everything to MLflow on DagsHub: params, metrics, model artifact, registry.
"""
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
)

from mlflow_config import init_mlflow
from features import build_features

EXPERIMENT_NAME = "scout_risk_classifier"
MODEL_NAME = "scout-risk-classifier"


def train():
    df, feature_cols = build_features()
    X, y = df[feature_cols], df["is_delayed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    params = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": 42,
    }

    init_mlflow(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="xgb_risk_classifier"):
        mlflow.log_params(params)

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds),
            "roc_auc": roc_auc_score(y_test, proba),
        }

        # --- confusion matrix: log raw counts as metrics (queryable across runs) ---
        cm = confusion_matrix(y_test, preds)
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "specificity": tn / (tn + fp) if (tn + fp) else 0.0,   # recall for the negative class
        })
        mlflow.log_metrics(metrics)
        print(metrics)

        # --- confusion matrix: log as a visual artifact too ---
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay(cm, display_labels=["on_time", "delayed"]).plot(ax=ax, cmap="Blues")
        plt.title("Risk Classifier — Confusion Matrix")
        plt.tight_layout()
        fig.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close(fig)

        mlflow.xgboost.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )


if __name__ == "__main__":
    train()