"""
SCOUT dashboard — reads model_predictions + shipments/suppliers from MySQL.

Usage: streamlit run dashboard/app.py
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from db_connect import get_sqlalchemy_engine

import streamlit as st
import pandas as pd

st.set_page_config(page_title="SCOUT — Supply Chain Risk Dashboard", layout="wide")

engine = get_sqlalchemy_engine()

@st.cache_data(ttl=60)
def load_predictions():
    query = """
        SELECT mp.prediction_id, mp.shipment_id, mp.risk_probability,
               mp.predicted_delay_days, mp.prediction_date,
               s.origin, s.destination, s.transport_mode, s.status,
               sup.name AS supplier_name, sup.region
        FROM model_predictions mp
        LEFT JOIN shipments s ON mp.shipment_id = s.shipment_id
        LEFT JOIN suppliers sup ON s.supplier_id = sup.supplier_id
        ORDER BY mp.prediction_date DESC
    """
    return pd.read_sql(query, engine)


st.title("🚨 SCOUT — Supply Chain Disruption Early Warning")

df = load_predictions()

if df.empty:
    st.warning("No predictions yet. Run `python agents/batch_run.py <n>` first.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Shipments Scored", len(df))
col2.metric("Avg Risk Probability", f"{df['risk_probability'].mean():.1%}")
col3.metric("High Risk (>60%)", int((df["risk_probability"] > 0.6).sum()))
col4.metric("Avg Predicted Delay", f"{df['predicted_delay_days'].mean():.2f} days")

st.subheader("Risk Distribution")
st.bar_chart(df["risk_probability"].value_counts(bins=10).sort_index())

st.subheader("Flagged Shipments (sorted by risk)")
min_risk = st.slider("Minimum risk probability", 0.0, 1.0, 0.0, 0.05)
filtered = df[df["risk_probability"] >= min_risk].sort_values("risk_probability", ascending=False)

st.dataframe(
    filtered[[
        "shipment_id", "supplier_name", "region", "origin", "destination",
        "transport_mode", "risk_probability", "predicted_delay_days", "prediction_date",
    ]],
    use_container_width=True,
    hide_index=True,
)

st.caption(f"Showing {len(filtered)} of {len(df)} scored shipments.")
