import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from db_connect import get_sqlalchemy_engine
from state import ScoutState

import pandas as pd
from sqlalchemy import text


class ShipmentNotFoundError(ValueError):
    pass


def monitor_agent(state: ScoutState) -> ScoutState:
    """Fetches shipment + supplier context for the shipment under review."""
    engine = get_sqlalchemy_engine()
    shipment_id = state["shipment_id"]

    query = text("""
        SELECT s.*, sup.name AS supplier_name, sup.country, sup.region,
               sup.category, sup.lead_time_days
        FROM shipments s
        LEFT JOIN suppliers sup ON s.supplier_id = sup.supplier_id
        WHERE s.shipment_id = :shipment_id
    """)
    df = pd.read_sql(query, engine, params={"shipment_id": shipment_id})
    print(f"[DEBUG] monitor_agent: shipment_id={shipment_id!r} (type={type(shipment_id)}), rows found={len(df)}")
    if df.empty:
        raise ShipmentNotFoundError(f"No shipment found with id {shipment_id}")

    state["shipment_info"] = df.iloc[0].to_dict()
    return state
