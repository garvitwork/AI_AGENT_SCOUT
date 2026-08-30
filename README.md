# SCOUT — Supply Chain Disruption Early-Warning Agent

An agentic AI system that monitors shipments, scores disruption risk and predicted delay using ML models tracked via MLflow/DagsHub, and generates plain-language, explainable recommendations using a multi-agent LangGraph pipeline powered by Gemini.

---

## Problem Statement

Enterprise supply-chain risk platforms (Blue Yonder, Kinaxis, o9) are built for large enterprises with six-figure budgets. Mid-market manufacturers and distributors — the segment that arguably needs early warning the most — are left relying on spreadsheets, gut feel, and reactive firefighting when shipments get delayed. There is no accessible, explainable, end-to-end system that combines live risk signals, predictive ML, and autonomous reasoning to flag at-risk shipments *before* they're late.

**SCOUT** addresses this gap: given a shipment, it estimates the probability of disruption, forecasts the expected delay, identifies the top contributing factors, and produces a human-readable recommendation — autonomously, with full audit logging.

---

## Approach

**Pipeline:**
```
Real + live data → MySQL → Feature Engineering → ML (MLflow/DagsHub) → Agentic reasoning (LangGraph + Gemini) → Dashboard
```

1. **Data layer** — Real-world DataCo Smart Supply Chain dataset (180K+ shipment records) loaded into MySQL as the historical backbone, plus a live-fetch pipeline (OpenWeather + NewsAPI) that continuously ingests current weather and news-based risk events per region.
2. **Feature engineering** — leak-free, time-aware features: scheduled transit days, order-date seasonality, a trailing-window count of nearby risk events, and each supplier's historical delay rate/magnitude computed with an expanding window (so no shipment ever "sees" its own or future outcomes).
3. **ML layer** — two XGBoost models: a disruption-risk classifier and a delay-duration regressor, both hyperparameter-tuned via `RandomizedSearchCV` and tracked in MLflow on DagsHub (params, metrics, confusion matrix, feature importance, model registry).
4. **Agentic layer** — a 3-agent LangGraph pipeline:
   - **MonitorAgent** — pulls shipment + supplier context from MySQL.
   - **RiskAgent** — loads the latest registered models from the MLflow registry, scores the shipment, and extracts top contributing factors.
   - **RecoAgent** — Gemini (`gemini-3.6-flash`) turns the scores into a concise, explainable, action-oriented recommendation.
5. **Batch + dashboard** — a batch runner scores many shipments at once (rate-limited for the free Gemini tier), and a Streamlit dashboard visualizes flagged shipments, risk distribution, and recommendations.

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data storage | MySQL |
| Data ingestion | pandas, requests (OpenWeather, NewsAPI) |
| ML | XGBoost, scikit-learn |
| MLOps | MLflow, DagsHub (experiment tracking + model registry) |
| Agent orchestration | LangGraph |
| LLM reasoning | Google Gemini (`gemini-3.6-flash`) via `langchain-google-genai` |
| Dashboard | Streamlit |

---

## Project Structure

```
AI_AGENT_SCOUT/
├── db_connect.py              # MySQL connection (pymysql + SQLAlchemy)
├── init_db.py                 # creates scout_db + tables from schema.sql
├── schema.sql                 # suppliers, shipments, risk_events, model_predictions
├── load_static_data.py        # ingests DataCo dataset into MySQL
├── fetch_live_signals.py      # live weather/news risk-event ingestion
├── requirements.txt
├── .env                       # DB creds, API keys (not committed)
│
├── data/raw/                  # DataCoSupplyChainDataset.csv + description
│
├── ml/
│   ├── mlflow_config.py       # MLflow tracking pointed at DagsHub
│   ├── features.py            # leak-free feature engineering
│   ├── train_risk_classifier.py
│   ├── train_delay_forecast.py
│   ├── confusion_matrix.png
│   └── delay_feature_importance.png
│
├── agents/
│   ├── state.py                # shared LangGraph state schema
│   ├── model_loader.py         # loads latest MLflow-registered models
│   ├── monitor_agent.py
│   ├── risk_agent.py
│   ├── reco_agent.py
│   ├── graph.py                # LangGraph wiring
│   ├── pipeline.py             # shared run + log logic
│   ├── main.py                 # single-shipment CLI runner
│   └── batch_run.py            # multi-shipment batch runner
│
└── batch_run/app.py           # Streamlit dashboard
```

---

## Setup & Run

```bash
pip install -r requirements.txt

# 1. Create DB + tables
python init_db.py

# 2. Load historical dataset
python load_static_data.py

# 3. Fetch live risk signals (run on a schedule in production)
python fetch_live_signals.py

# 4. Train models (logged to MLflow/DagsHub)
cd ml
python train_risk_classifier.py
python train_delay_forecast.py

# 5. Run the agent pipeline
cd ../agents
python main.py <shipment_id>          # single shipment
python batch_run.py 30                # batch of 30 shipments

# 6. View results
cd ..
streamlit run batch_run/app.py
```

---

## Challenges Faced & How They Were Solved

**1. Silent data loss from a broken aggregation (180K → 187 rows).**
An early `groupby().transform()` chain in the ingestion script (`.transform("mean")` followed by `.drop_duplicates().reindex()`) misaligned indices and silently produced `NaN` for nearly every supplier's `lead_time_days`. Since the feature pipeline drops rows with missing values, 99.9% of the dataset was invisible to training without any error being raised. Diagnosed by explicitly printing row counts after feature-building, then fixed by replacing the transform chain with a clean `groupby().mean()` mapped back onto suppliers by region.

**2. Unlinked foreign keys (`shipments.supplier_id` always NULL).**
The original ingestion script inserted `shipments` without ever assigning `supplier_id`, so every merge against `suppliers` failed with a dtype mismatch (`object` vs `int64`) — masking the real issue. Fixed by reading back the auto-generated `supplier_id`s after inserting suppliers and mapping them onto shipments via a natural key (region + category) before insert.

**3. MySQL password containing `@` breaking connection strings.**
A raw password like `garvit@123` is safe as a `pymysql.connect()` keyword argument but breaks a SQLAlchemy URI string (the `@` gets misread as the host separator). Solved by using `urllib.parse.quote_plus()` to percent-encode the password specifically for URI-based connections, while keeping the raw kwarg for direct `pymysql` connections.

**4. Live-fetch script failing silently.**
`fetch_live_signals.py` initially never called `load_dotenv()`, so API keys resolved to empty strings and every request failed with a 401 — but the script only printed "No events to insert" with no indication why. Fixed by loading `.env` explicitly and surfacing the actual HTTP status/error text instead of failing quietly.

**5. Hyperparameter tuning plateau on the delay regressor.**
After adding new features and running a 25-candidate randomized search, R² barely moved (0.278 → 0.278). Rather than continuing to tune blindly, the plateau was diagnosed as a genuine signal ceiling: the added `scheduled_days` feature was almost fully collinear with `transport_mode` (already in the model), and the target itself carries irreducible noise in this dataset — consistent with public benchmarks on the same data. Recognizing "no more signal to extract" (vs. "wrong hyperparameters") avoided wasted tuning cycles.

**6. Rapidly deprecating Gemini model identifiers.**
`gemini-1.5-flash` → `gemini-2.5-flash` → `gemini-3.6-flash`, each retired within the build window. Solved by treating the model name as a single-point config value and reading the API's own error message (which named the recommended replacement) rather than guessing.

**7. Nested/misaligned project paths (Windows).**
Terminal `cd` mistakes (running from a duplicated nested folder, or from the wrong subdirectory) caused repeated `ModuleNotFoundError` and `FileNotFoundError`. Solved by standardizing on relative imports anchored to `os.path.dirname(__file__)` and always running scripts from their intended working directory.

---

## Results

| Model | Metric | Value |
|---|---|---|
| Risk classifier | Accuracy | 70.2% |
| Risk classifier | ROC-AUC | 0.757 |
| Risk classifier | Precision / Recall | 0.87 / 0.57 |
| Delay forecaster | MAE | 0.98 days |
| Delay forecaster | R² | 0.28 |

Both models are versioned in the MLflow model registry on DagsHub (4 iterations each), with confusion matrix and feature-importance artifacts logged per run.

---

## Conclusion & Key Takeaways

- **Debugging discipline mattered more than model choice.** The two biggest performance jumps (187 → 180,519 rows; 57.9% → 70.3% accuracy) came from fixing silent data-pipeline bugs, not from tuning or trying new algorithms. A correct-looking script can still destroy most of a dataset without raising an error — row-count sanity checks after every transform are non-negotiable.
- **Leak-free feature engineering is a discipline, not an afterthought.** Every "supplier history" feature was deliberately computed as an expanding window shifted by one, specifically to prevent a shipment's own outcome from leaking into its own prediction — a subtlety that's easy to get wrong silently.
- **Knowing when to stop tuning is itself a skill.** Recognizing a genuine data ceiling (via a flat hyperparameter search) prevented burning further cycles chasing marginal gains that weren't there, and reframed the result as a legitimate, defensible v1 rather than an unfinished one.
- **Explainability was built in, not bolted on.** Every RiskAgent output carries its top contributing features, and every recommendation is grounded in those numbers before Gemini writes a single sentence — critical for a system meant to be trusted by a supply chain manager.
- **End-to-end ownership**: this project spans data engineering, MLOps, ML modeling, and agentic orchestration — the full stack a modern ML/AI engineering role expects, on a real, underserved business problem rather than a toy dataset.

---

## Future Work

- Migrate `RandomizedSearchCV` to Bayesian optimization (Optuna) for more efficient search.
- Add SHAP-based explanations to replace the current feature-importance proxy.
- Wrap the pipeline in FastAPI for production-style serving.
- Add automated retraining triggers (drift monitoring) as new live risk events accumulate.
- Extend live-fetch to port-congestion and customs-delay data sources.