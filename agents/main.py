import sys
from pipeline import run_shipment


def run(shipment_id: int):
    result = run_shipment(shipment_id)

    print(f"\n--- SCOUT Report: Shipment {shipment_id} ---")
    print(f"Risk probability: {result['risk_probability']}")
    print(f"Predicted delay: {result['predicted_delay_days']} days")
    print(f"Top risk factors: {result['top_risk_factors']}")
    print(f"\nRecommendation:\n{result['recommendation']}")
    print("\nLogged to model_predictions.")


if __name__ == "__main__":
    shipment_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    run(shipment_id)
