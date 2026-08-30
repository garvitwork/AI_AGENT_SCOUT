"""
SCOUT FastAPI server. Run from agents/ folder:
  uvicorn api:app --host 0.0.0.0 --port 8000

On Render, set start command to:
  uvicorn api:app --host 0.0.0.0 --port $PORT
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from db_connect import get_sqlalchemy_engine

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

from pipeline import run_shipment

app = FastAPI(title="SCOUT API", version="1.0")

# open CORS for now — restrict to your frontend's domain once it exists
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict/{shipment_id}")
def predict(shipment_id: int):
    try:
        result = run_shipment(shipment_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    info = result.get("shipment_info", {})
    return {
        "shipment_id": shipment_id,
        "origin": info.get("origin"),
        "destination": info.get("destination"),
        "supplier_name": info.get("supplier_name"),
        "transport_mode": info.get("transport_mode"),
        "risk_probability": result["risk_probability"],
        "predicted_delay_days": result["predicted_delay_days"],
        "top_risk_factors": result["top_risk_factors"],
        "recommendation": result["recommendation"],
    }


@app.get("/predictions")
def list_predictions(limit: int = 50):
    engine = get_sqlalchemy_engine()
    query = f"""
        SELECT mp.prediction_id, mp.shipment_id, mp.risk_probability,
               mp.predicted_delay_days, mp.prediction_date,
               s.origin, s.destination, s.transport_mode,
               sup.name AS supplier_name
        FROM model_predictions mp
        LEFT JOIN shipments s ON mp.shipment_id = s.shipment_id
        LEFT JOIN suppliers sup ON s.supplier_id = sup.supplier_id
        ORDER BY mp.prediction_date DESC
        LIMIT {limit}
    """
    df = pd.read_sql(query, engine)
    return df.to_dict(orient="records")