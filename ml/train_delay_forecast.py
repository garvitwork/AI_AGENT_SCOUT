"""
Trains the delay-duration regressor (predicted_delay_days) with
hyperparameter search, logs everything to MLflow on DagsHub.
"""
import mlflow
import mlflow.xgboost
import matplotlib.pyplot as plt
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from mlflow_config import init_mlflow
from features import build_features

EXPERIMENT_NAME = "scout_delay_forecast"
MODEL_NAME = "scout-delay-forecaster"

PARAM_GRID = {
    "n_estimators": [200, 300, 500, 800],
    "max_depth": [3, 4, 5, 6, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.6, 0.7, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.7, 0.8, 1.0],
    "min_child_weight": [1, 3, 5],
    "reg_alpha": [0, 0.1, 0.5],
    "reg_lambda": [1, 1.5, 2],
}


def train():
    df, feature_cols = build_features()
    X, y = df[feature_cols], df["delay_days"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    init_mlflow(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="xgb_delay_forecaster_tuned"):

        search = RandomizedSearchCV(
            estimator=XGBRegressor(random_state=42),
            param_distributions=PARAM_GRID,
            n_iter=25,
            scoring="neg_mean_absolute_error",
            cv=3,
            n_jobs=-1,
            random_state=42,
            verbose=1,
        )
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        mlflow.log_params(search.best_params_)
        mlflow.log_metric("cv_best_neg_mae", search.best_score_)

        preds = best_model.predict(X_test)
        metrics = {
            "mae": mean_absolute_error(y_test, preds),
            "rmse": mean_squared_error(y_test, preds) ** 0.5,
            "r2": r2_score(y_test, preds),
        }
        mlflow.log_metrics(metrics)
        print("Best params:", search.best_params_)
        print("Test metrics:", metrics)

        # --- feature importance artifact ---
        importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values()
        fig, ax = plt.subplots(figsize=(6, 5))
        importances.plot(kind="barh", ax=ax)
        plt.title("Delay Forecaster — Feature Importance")
        plt.tight_layout()
        fig.savefig("delay_feature_importance.png")
        mlflow.log_artifact("delay_feature_importance.png")
        plt.close(fig)

        mlflow.xgboost.log_model(
            best_model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )


if __name__ == "__main__":
    train()