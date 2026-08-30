"""
Loads the DataCo Smart Supply Chain dataset (static, real-world) into MySQL.

Dataset: "DataCo Smart Supply Chain for Big Data Analysis" (Kaggle)
  https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis
Download DataCoSupplyChainDataset.csv manually (Kaggle needs auth) and place it at:
  scout/data/raw/DataCoSupplyChainDataset.csv

Usage: python data/scripts/load_static_data.py
"""


import pandas as pd
import os
from db_connect import get_sqlalchemy_engine

RAW_PATH = os.path.join(os.path.dirname(__file__), "data", "raw", "DataCoSupplyChainDataset.csv")


def load_and_transform():
    df = pd.read_csv(RAW_PATH, encoding="latin1")

    # --- suppliers table (derived from unique supplier/market info) ---
    suppliers = (
        df[["Order Region", "Market", "Category Name"]]
        .drop_duplicates()
        .rename(columns={
            "Order Region": "region",
            "Market": "country",
            "Category Name": "category",
        })
    )
    suppliers["name"] = suppliers["region"] + " - " + suppliers["category"]
    region_lead_time = df.groupby("Order Region")["Days for shipping (real)"].mean().round().astype(int)
    suppliers["lead_time_days"] = suppliers["region"].map(region_lead_time)
    suppliers = suppliers[["name", "country", "region", "category", "lead_time_days"]]

    # --- shipments table ---
    shipments = pd.DataFrame({
        "order_date": pd.to_datetime(df["order date (DateOrders)"], errors="coerce"),
        "expected_delivery_date": pd.to_datetime(df["order date (DateOrders)"], errors="coerce") +
                                   pd.to_timedelta(df["Days for shipment (scheduled)"], unit="D"),
        "actual_delivery_date": pd.to_datetime(df["order date (DateOrders)"], errors="coerce") +
                                 pd.to_timedelta(df["Days for shipping (real)"], unit="D"),
        "status": df["Delivery Status"],
        "quantity": df["Order Item Quantity"],
        "product_category": df["Category Name"],
        "origin": df["Order Region"],
        "destination": df["Order Country"],
        "transport_mode": df["Shipping Mode"],
        "delay_days": df["Days for shipping (real)"] - df["Days for shipment (scheduled)"],
    })

    engine = get_sqlalchemy_engine()
    suppliers.to_sql("suppliers", engine, if_exists="append", index=False)

    db_suppliers = pd.read_sql("SELECT supplier_id, region, category FROM suppliers", engine)
    df = df.merge(
        db_suppliers,
        left_on=["Order Region", "Category Name"],
        right_on=["region", "category"],
        how="left",
    )
    shipments["supplier_id"] = df["supplier_id"]

    shipments.to_sql("shipments", engine, if_exists="append", index=False)
    print(f"Loaded {len(suppliers)} suppliers and {len(shipments)} shipments into scout_db.")

if __name__ == "__main__":
    load_and_transform()
