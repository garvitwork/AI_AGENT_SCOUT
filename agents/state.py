from typing import TypedDict, Optional


class ScoutState(TypedDict, total=False):
    shipment_id: int
    shipment_info: dict          # raw shipment + supplier context (MonitorAgent)
    risk_probability: float      # RiskAgent
    predicted_delay_days: float  # RiskAgent
    top_risk_factors: list       # RiskAgent (feature importance based)
    recommendation: str          # RecoAgent
