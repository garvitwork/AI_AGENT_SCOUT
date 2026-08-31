"""
Computes the ML feature vector for ONE shipment using targeted SQL
queries (COUNT/AVG aggregates), instead of loading the entire 180K-row
dataset just to score a single shipment. This is what fixed the
out-of-memory crash on Render's 512MB tier.

Reuses shipment_info already fetched by MonitorAgent — no extra query
for the shipment itself.
"""
import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from db_connect import get_sqlalchemy_engine

import pandas as pd
from sqlalchemy import text

FEATURE_COLS = [
    "quantity", "lead_time_days", "region_risk_event_count", "scheduled_days",
    "order_month", "order_dayofweek", "order_quarter",
    "supplier_historical_delay_rate", "supplier_historical_avg_delay",
    "product_category", "origin", "destination", "transport_mode", "category",
]

RISK_WINDOW_DAYS = 14

_mappings = None


def _load_mappings():
    global _mappings
    if _mappings is None:
        path = os.path.join(os.path.dirname(__file__), "..", "outputs", "category_mappings.json")
        with open(path) as f:
            _mappings = json.load(f)
    return _mappings


def _encode(col: str, value) -> int:
    mapping = _load_mappings().get(col, {})
    return mapping.get(str(value), -1)  # -1 for unseen category, matches pandas' NaN-code convention


def build_single_shipment_features(shipment_info: dict) -> pd.DataFrame:
    engine = get_sqlalchemy_engine()

    order_date = pd.to_datetime(shipment_info["order_date"])
    expected_delivery_date = pd.to_datetime(shipment_info["expected_delivery_date"])
    supplier_id = shipment_info["supplier_id"]
    origin = shipment_info["origin"]

    # --- region_risk_event_count: trailing-window COUNT, not a full-table scan ---
    q_events = text("""
        SELECT COUNT(*) AS cnt FROM risk_events
        WHERE region = :region
          AND event_date > :lo AND event_date <= :hi
    """)
    lo = order_date - pd.Timedelta(days=RISK_WINDOW_DAYS)
    with engine.connect() as conn:
        region_risk_event_count = conn.execute(
            q_events, {"region": origin, "lo": lo, "hi": order_date}
        ).scalar() or 0

    # --- supplier historical delay rate/magnitude: aggregate over PRIOR orders only ---
    q_hist = text("""
        SELECT AVG(CASE WHEN delay_days > 0 THEN 1.0 ELSE 0.0 END) AS delay_rate,
               AVG(delay_days) AS avg_delay
        FROM shipments
        WHERE supplier_id = :supplier_id AND order_date < :order_date
    """)
    with engine.connect() as conn:
        row = conn.execute(q_hist, {"supplier_id": supplier_id, "order_date": order_date}).mappings().first()

    supplier_historical_delay_rate = row["delay_rate"] if row and row["delay_rate"] is not None else 0.573  # global fallback
    supplier_historical_avg_delay = row["avg_delay"] if row and row["avg_delay"] is not None else 0.0

    features = {
        "quantity": shipment_info["quantity"],
        "lead_time_days": shipment_info["lead_time_days"],
        "region_risk_event_count": region_risk_event_count,
        "scheduled_days": (expected_delivery_date - order_date).days,
        "order_month": order_date.month,
        "order_dayofweek": order_date.dayofweek,
        "order_quarter": order_date.quarter,
        "supplier_historical_delay_rate": supplier_historical_delay_rate,
        "supplier_historical_avg_delay": supplier_historical_avg_delay,
        "product_category": _encode("product_category", shipment_info["product_category"]),
        "origin": _encode("origin", shipment_info["origin"]),
        "destination": _encode("destination", shipment_info["destination"]),
        "transport_mode": _encode("transport_mode", shipment_info["transport_mode"]),
        "category": _encode("category", shipment_info["category"]),
    }

    return pd.DataFrame([features])[FEATURE_COLS]
