"""
Builds ML-ready features from MySQL for both the risk classifier and
delay regressor. v2: adds time features, date-aligned risk-event counts,
and a leak-free supplier historical delay rate.
"""
import numpy as np
import pandas as pd
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from db_connect import get_sqlalchemy_engine


def _count_events_in_window(order_dates: pd.Series, event_dates: np.ndarray, window_days: int) -> np.ndarray:
    """For each order_date, count how many event_dates fall in
    (order_date - window_days, order_date] — a proper 'recent risk' signal."""
    event_dates = np.sort(event_dates)
    order_ts = order_dates.values.astype("datetime64[ns]")
    lo = order_ts - np.timedelta64(window_days, "D")
    hi = order_ts
    idx_lo = np.searchsorted(event_dates, lo, side="right")
    idx_hi = np.searchsorted(event_dates, hi, side="right")
    return idx_hi - idx_lo


def build_features(risk_window_days: int = 14):
    engine = get_sqlalchemy_engine()

    shipments = pd.read_sql("SELECT * FROM shipments", engine)
    suppliers = pd.read_sql("SELECT * FROM suppliers", engine)
    risk_events = pd.read_sql("SELECT * FROM risk_events", engine)

    df = shipments.merge(suppliers, on="supplier_id", how="left", suffixes=("", "_supplier"))
    df["order_date"] = pd.to_datetime(df["order_date"])
    df = df.dropna(subset=["order_date"]).sort_values("order_date").reset_index(drop=True)

    # --- target ---
    df["is_delayed"] = (df["delay_days"] > 0).astype(int)

    # --- scheduled transit days: the committed delivery window at order time — strong delay predictor ---
    df["expected_delivery_date"] = pd.to_datetime(df["expected_delivery_date"])
    df["scheduled_days"] = (df["expected_delivery_date"] - df["order_date"]).dt.days

    # --- time features ---
    df["order_month"] = df["order_date"].dt.month
    df["order_dayofweek"] = df["order_date"].dt.dayofweek
    df["order_quarter"] = df["order_date"].dt.quarter

    # --- date-aligned risk-event count per region, trailing window (real signal, not a flat count) ---
    risk_events["event_date"] = pd.to_datetime(risk_events["event_date"])
    df["region_risk_event_count"] = 0
    for region, group_idx in df.groupby("origin").groups.items():
        region_events = risk_events.loc[risk_events["region"] == region, "event_date"].values
        if len(region_events) == 0:
            continue
        idx = df.loc[group_idx].index
        counts = _count_events_in_window(df.loc[idx, "order_date"], region_events, risk_window_days)
        df.loc[idx, "region_risk_event_count"] = counts

    # --- supplier historical delay rate (leak-free: only prior orders, via expanding mean shifted by 1) ---
    df["supplier_historical_delay_rate"] = (
        df.groupby("supplier_id")["is_delayed"]
        .apply(lambda s: s.expanding().mean().shift(1))
        .reset_index(level=0, drop=True)
    )
    # first order per supplier has no history yet — backfill with global mean
    df["supplier_historical_delay_rate"] = df["supplier_historical_delay_rate"].fillna(
        df["is_delayed"].mean()
    )

    # --- supplier historical avg delay magnitude (leak-free, for the regressor) ---
    df["supplier_historical_avg_delay"] = (
        df.groupby("supplier_id")["delay_days"]
        .apply(lambda s: s.expanding().mean().shift(1))
        .reset_index(level=0, drop=True)
    )
    df["supplier_historical_avg_delay"] = df["supplier_historical_avg_delay"].fillna(
        df["delay_days"].mean()
    )

    # --- categorical encoding ---
    for col in ["status", "product_category", "origin", "destination", "transport_mode", "category"]:
        df[col] = df[col].astype("category").cat.codes

    feature_cols = [
        "quantity", "lead_time_days", "region_risk_event_count", "scheduled_days",
        "order_month", "order_dayofweek", "order_quarter",
        "supplier_historical_delay_rate", "supplier_historical_avg_delay",
        "product_category", "origin", "destination", "transport_mode", "category",
    ]
    df = df.dropna(subset=feature_cols + ["delay_days", "is_delayed"])

    return df, feature_cols


if __name__ == "__main__":
    df, cols = build_features()
    print(df[cols + ["is_delayed", "delay_days"]].head())
    print(f"\nRows: {len(df)}")
    print(f"Delayed rate: {df['is_delayed'].mean():.3f}")