import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

from state import ScoutState

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.3,
)

PROMPT_TEMPLATE = """You are a supply chain risk advisor. Given the data below,
write a concise (3-4 sentence) recommendation for a supply chain manager.
State clearly: (1) the risk level in plain language, (2) the expected delay,
(3) the top factor(s) driving this risk, (4) one concrete recommended action
(e.g. expedite, reroute, contact supplier, no action needed).

Shipment: {shipment_id} | Supplier: {supplier_name} | Route: {origin} -> {destination}
Transport mode: {transport_mode}
Disruption risk probability: {risk_probability}
Predicted delay: {predicted_delay_days} days
Top contributing factors: {top_risk_factors}
"""


def reco_agent(state: ScoutState) -> ScoutState:
    info = state["shipment_info"]
    prompt = PROMPT_TEMPLATE.format(
        shipment_id=state["shipment_id"],
        supplier_name=info.get("supplier_name", "unknown"),
        origin=info.get("origin", "?"),
        destination=info.get("destination", "?"),
        transport_mode=info.get("transport_mode", "?"),
        risk_probability=state["risk_probability"],
        predicted_delay_days=state["predicted_delay_days"],
        top_risk_factors=state["top_risk_factors"],
    )
    response = llm.invoke(prompt)
    state["recommendation"] = response.content
    return state
