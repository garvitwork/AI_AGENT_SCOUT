from langgraph.graph import StateGraph, END

from state import ScoutState
from monitor_agent import monitor_agent
from risk_agent import risk_agent
from reco_agent import reco_agent


def build_graph():
    graph = StateGraph(ScoutState)

    graph.add_node("monitor", monitor_agent)
    graph.add_node("risk", risk_agent)
    graph.add_node("recommend", reco_agent)

    graph.set_entry_point("monitor")
    graph.add_edge("monitor", "risk")
    graph.add_edge("risk", "recommend")
    graph.add_edge("recommend", END)

    return graph.compile()
