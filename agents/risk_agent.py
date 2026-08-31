import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "ml"))
from features import build_features

from state import ScoutState
from model_loader import load_models

_risk_model = None
_delay_model = None
_features_df = None
_feature_cols = None


def _get_models():
    global _risk_model, _delay_model
    if _risk_model is None:
        _risk_model, _delay_model = load_models()
    return _risk_model, _delay_model


def _get_features():
    global _features_df, _feature_cols
    if _features_df is None:
        _features_df, _feature_cols = build_features()
    return _features_df, _feature_cols


def risk_agent(state: ScoutState) -> ScoutState:
    """Scores the shipment's disruption risk and predicted delay using
    the latest MLflow-registered models, plus a simple feature-importance
    based explanation (top contributing factors)."""
    risk_model, delay_model = _get_models()
    df, feature_cols = _get_features()
    row = df[df["shipment_id"] == state["shipment_id"]]
    if row.empty:
        raise ValueError(f"shipment_id {state['shipment_id']} not found in feature set")

    X = row[feature_cols]

    risk_probability = float(risk_model.predict_proba(X)[0, 1])
    predicted_delay_days = float(delay_model.predict(X)[0])

    # simple explainability: top 3 features by (importance * value) contribution proxy
    importances = risk_model.feature_importances_
    contrib = sorted(
        zip(feature_cols, importances, X.iloc[0].values),
        key=lambda t: t[1],
        reverse=True,
    )[:3]
    top_risk_factors = [{"feature": f, "importance": float(i), "value": float(v)} for f, i, v in contrib]

    state["risk_probability"] = round(risk_probability, 4)
    state["predicted_delay_days"] = round(predicted_delay_days, 2)
    state["top_risk_factors"] = top_risk_factors
    return state