import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from db_connect import get_pymysql_connection

from graph import build_graph

_app = None


def _get_app():
    global _app
    if _app is None:
        _app = build_graph()
    return _app


def log_prediction(shipment_id, risk_probability, predicted_delay_days):
    conn = get_pymysql_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO model_predictions
                (shipment_id, model_name, model_version, risk_probability, predicted_delay_days)
            VALUES (%s, %s, %s, %s, %s)
            """,
            (shipment_id, "scout-risk-classifier+scout-delay-forecaster", "latest",
             risk_probability, predicted_delay_days),
        )
    conn.close()


def run_shipment(shipment_id: int) -> dict:
    """Runs the full MonitorAgent -> RiskAgent -> RecoAgent pipeline for one
    shipment and logs the prediction. Returns the result dict."""
    app = _get_app()
    result = app.invoke({"shipment_id": shipment_id})
    log_prediction(shipment_id, result["risk_probability"], result["predicted_delay_days"])
    return result
