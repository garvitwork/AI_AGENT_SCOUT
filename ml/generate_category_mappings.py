"""
Generates the same categorical -> integer code mapping that pandas'
`.astype("category").cat.codes` produced during training (pandas assigns
codes 0..n-1 in sorted order by default) — but via lightweight SQL
DISTINCT queries instead of loading the full 180K-row dataset.

Run this once after training (or whenever the underlying categories
could have changed) so the API can encode a single shipment's
categorical fields at inference time without loading the whole table.

Usage: python generate_category_mappings.py
"""
import sys, os, json
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from db_connect import get_sqlalchemy_engine

import pandas as pd

# column -> (source table, source column)
CATEGORICAL_SOURCES = {
    "status": ("shipments", "status"),
    "product_category": ("shipments", "product_category"),
    "origin": ("shipments", "origin"),
    "destination": ("shipments", "destination"),
    "transport_mode": ("shipments", "transport_mode"),
    "category": ("suppliers", "category"),
}


def generate_mappings():
    engine = get_sqlalchemy_engine()
    mappings = {}

    for col, (table, source_col) in CATEGORICAL_SOURCES.items():
        query = f"SELECT DISTINCT {source_col} FROM {table} WHERE {source_col} IS NOT NULL ORDER BY {source_col} ASC"
        values = pd.read_sql(query, engine)[source_col].tolist()
        mappings[col] = {str(v): i for i, v in enumerate(values)}

    out_path = os.path.join(os.path.dirname(__file__), "..", "outputs", "category_mappings.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(mappings, f, indent=2)

    print(f"Saved category mappings for {list(mappings.keys())} to {out_path}")


if __name__ == "__main__":
    generate_mappings()
