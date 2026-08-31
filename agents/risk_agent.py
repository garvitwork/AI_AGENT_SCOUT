from state import ScoutState
from model_loader import load_models
from inference_features import build_single_shipment_features, FEATURE_COLS

_risk_model = None
_delay_model = None


def _get_models():
    global _risk_model, _delay_model
    if _risk_model is None:
        _risk_model, _delay_model = load_models()
    return _risk_model, _delay_model


def risk_agent(state: ScoutState) -> ScoutState:
    """Scores the shipment's disruption risk and predicted delay using
    the latest MLflow-registered models, plus a simple feature-importance
    based explanation (top contributing factors). Uses a lightweight,
    single-shipment SQL-based feature build — not the full training
    dataset — to keep memory usage low in production."""
    risk_model, delay_model = _get_models()

    X = build_single_shipment_features(state["shipment_info"])

    risk_probability = float(risk_model.predict_proba(X)[0, 1])
    predicted_delay_days = float(delay_model.predict(X)[0])

    importances = risk_model.feature_importances_
    contrib = sorted(
        zip(FEATURE_COLS, importances, X.iloc[0].values),
        key=lambda t: t[1],
        reverse=True,
    )[:3]
    top_risk_factors = [{"feature": f, "importance": float(i), "value": float(v)} for f, i, v in contrib]

    state["risk_probability"] = round(risk_probability, 4)
    state["predicted_delay_days"] = round(predicted_delay_days, 2)
    state["top_risk_factors"] = top_risk_factors
    return state