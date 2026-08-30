"""
Runs the SCOUT pipeline across multiple shipments in one go.

Usage:
  python batch_run.py 50          # 50 random shipments
  python batch_run.py 50 --recent # 50 most recent shipments instead of random
"""
import sys, os, time, argparse
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from db_connect import get_sqlalchemy_engine

import pandas as pd
from tqdm import tqdm
from pipeline import run_shipment

GEMINI_CALL_DELAY_SECONDS = 4  # free-tier Gemini rate limit safety margin


def get_shipment_ids(n: int, recent: bool) -> list[int]:
    engine = get_sqlalchemy_engine()
    order_clause = "ORDER BY order_date DESC" if recent else "ORDER BY RAND()"
    df = pd.read_sql(f"SELECT shipment_id FROM shipments {order_clause} LIMIT {n}", engine)
    return df["shipment_id"].tolist()


def batch_run(n: int, recent: bool = False):
    shipment_ids = get_shipment_ids(n, recent)
    print(f"Running SCOUT on {len(shipment_ids)} shipments...")

    results, errors = [], []
    for sid in tqdm(shipment_ids):
        try:
            result = run_shipment(sid)
            results.append({
                "shipment_id": sid,
                "risk_probability": result["risk_probability"],
                "predicted_delay_days": result["predicted_delay_days"],
            })
        except Exception as e:
            errors.append({"shipment_id": sid, "error": str(e)})
        time.sleep(GEMINI_CALL_DELAY_SECONDS)

    print(f"\nDone. Success: {len(results)}, Failed: {len(errors)}")
    if errors:
        print("Failures:", errors[:5], "..." if len(errors) > 5 else "")

    return results, errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int, help="number of shipments to process")
    parser.add_argument("--recent", action="store_true", help="use most recent shipments instead of random")
    args = parser.parse_args()

    batch_run(args.n, args.recent)
