# SCOUT — Supply Chain Disruption Early-Warning Agent

An agentic AI system that monitors shipments, scores disruption risk and predicted delay using ML models tracked via MLflow/DagsHub, and generates plain-language, explainable recommendations using a multi-agent LangGraph pipeline powered by Gemini — deployed end-to-end as a live API and frontend.

**Live demo:** [garvitwork.github.io/frontend_scout](https://garvitwork.github.io/frontend_scout/)
**Backend repo:** [github.com/garvitwork/AI_AGENT_SCOUT](https://github.com/garvitwork/AI_AGENT_SCOUT)
**API:** [ai-agent-scout.onrender.com](https://ai-agent-scout.onrender.com) — interactive docs at `/docs`
**Experiment tracking:** DagsHub (MLflow tracking + model registry + DVC pipeline)

---

## Problem Statement

Enterprise supply-chain risk platforms (Blue Yonder, Kinaxis, o9) are built for large enterprises with six-figure budgets. Mid-market manufacturers and distributors — the segment that arguably needs early warning the most — are left relying on spreadsheets, gut feel, and reactive firefighting when shipments get delayed. There is no accessible, explainable, end-to-end system that combines live risk signals, predictive ML, and autonomous reasoning to flag at-risk shipments *before* they're late.

**SCOUT** addresses this gap: given a shipment, it estimates the probability of disruption, forecasts the expected delay, identifies the top contributing factors, and produces a human-readable recommendation — autonomously, with full audit logging, served through a live API and dashboard.

---

## Approach

**Pipeline:**
```
Real + live data → Cloud MySQL (Aiven) → DVC-tracked feature/training pipeline
→ ML (MLflow/DagsHub registry) → Agentic reasoning (LangGraph + Gemini)
→ FastAPI (Render) → Frontend (GitHub Pages)
```

1. **Data layer** — Real-world DataCo Smart Supply Chain dataset (180K+ shipment records) loaded into a cloud MySQL instance (Aiven, SSL-enforced), plus a live-fetch pipeline (OpenWeather + NewsAPI) that continuously ingests current weather and news-based risk events per region.
2. **Feature engineering** — leak-free, time-aware features: scheduled transit days, order-date seasonality, a trailing-window count of nearby risk events, and each supplier's historical delay rate/magnitude computed with an expanding window (so no shipment ever "sees" its own or future outcomes).
3. **ML layer** — two XGBoost models: a disruption-risk classifier and a delay-duration regressor, both hyperparameter-tuned via `RandomizedSearchCV` and tracked in MLflow on DagsHub (params, metrics, confusion matrix, feature importance, model registry).
4. **Pipeline orchestration (DVC)** — the full path from raw data to trained model is defined as a `dvc.yaml` DAG (ingest → build features → train risk classifier → train delay forecaster), with hyperparameters centralized in `params.yaml` and metrics/plots versioned as DVC artifacts, visible on DagsHub's Data Pipeline and Experiments tabs.
5. **Agentic layer** — a 3-agent LangGraph pipeline:
   - **MonitorAgent** — pulls shipment + supplier context from MySQL.
   - **RiskAgent** — loads the latest registered models from the MLflow registry, scores the shipment using a **lightweight, single-shipment SQL feature build** (not the full training dataset — critical for running in a memory-constrained cloud environment), and extracts top contributing factors.
   - **RecoAgent** — Gemini (`gemini-3.6-flash`) turns the scores into a concise, explainable, action-oriented recommendation.
6. **Serving** — a FastAPI backend (`/predict/{shipment_id}`, `/predictions`, `/health`) deployed on Render, plus a batch runner and a Streamlit ops dashboard for internal use.
7. **Frontend** — a standalone HTML/CSS/JS "control tower" dashboard (dark radar theme, boarding-pass-style result ticket, live departure board) hosted on GitHub Pages, calling the Render API directly.

---

## Tech Stack

| Category | Tools | Why it's here |
|---|---|---|
| **Cloud Database** | MySQL (Aiven, cloud-hosted, SSL-enforced) | Production-grade relational storage — not local/SQLite — reachable from any deployment target, TLS-secured end to end |
| **Data Ingestion** | pandas, requests (OpenWeather API, NewsAPI) | Blends a real historical dataset with live, continuously-refreshed risk signals |
| **Machine Learning** | XGBoost, scikit-learn | Gradient-boosted classification + regression, hyperparameter-tuned via `RandomizedSearchCV` |
| **MLOps** | MLflow, DVC, DagsHub | Full experiment tracking, model registry, and a versioned, reproducible pipeline DAG (`dvc repro`) |
| **Agent Orchestration** | LangGraph | Stateful multi-agent workflow — Monitor → Risk → Recommend |
| **LLM Reasoning** | Google Gemini (`gemini-3.6-flash`) via `langchain-google-genai` | Turns model outputs into explainable, human-readable recommendations |
| **Backend API** | FastAPI, deployed on Render | Production REST API serving the agent pipeline over HTTPS |
| **Frontend** | Vanilla HTML/CSS/JS, hosted on GitHub Pages | Custom-designed live dashboard, zero framework overhead |
| **Ops Dashboard** | Streamlit | Internal batch-monitoring view |
| **Version Control** | Git, dual-remote (GitHub + DagsHub) | Code and ML artifacts tracked and pushed together |

---

## Project Structure

```
AI_AGENT_SCOUT/
├── db_connect.py              # MySQL connection (pymysql + SQLAlchemy, SSL-aware)
├── init_db.py                 # creates tables from schema.sql
├── schema.sql                 # suppliers, shipments, risk_events, model_predictions
├── load_static_data.py        # ingests DataCo dataset into MySQL
├── fetch_live_signals.py      # live weather/news risk-event ingestion
├── ca.pem                     # Aiven CA certificate (public, safe to commit)
├── dvc.yaml                   # DVC pipeline DAG
├── params.yaml                # centralized hyperparameters, tracked by DVC
├── requirements.txt
├── .python-version             # pins Python 3.10/3.11 for cloud builds
├── .env                        # DB creds, API keys (not committed)
│
├── data/raw/                  # DataCoSupplyChainDataset.csv + description
│
├── ml/
│   ├── mlflow_config.py        # MLflow tracking pointed at DagsHub
│   ├── features.py             # leak-free feature engineering (training-time, full dataset)
│   ├── generate_category_mappings.py  # lightweight SQL-based category encoding for inference
│   ├── train_risk_classifier.py
│   ├── train_delay_forecast.py
│   └── outputs/ (confusion_matrix.png, delay_feature_importance.png, *_metrics.json, category_mappings.json)
│
├── agents/
│   ├── state.py                 # shared LangGraph state schema
│   ├── model_loader.py          # loads latest MLflow-registered models
│   ├── monitor_agent.py
│   ├── inference_features.py    # lightweight single-shipment feature build (production-safe)
│   ├── risk_agent.py
│   ├── reco_agent.py
│   ├── graph.py                 # LangGraph wiring
│   ├── pipeline.py              # shared run + log logic
│   ├── main.py                  # single-shipment CLI runner
│   ├── batch_run.py             # multi-shipment batch runner
│   └── api.py                   # FastAPI app (deployed on Render)
│
├── dashboard/app.py             # Streamlit ops dashboard
└── frontend_scout/              # standalone frontend (GitHub Pages)
    ├── index.html
    ├── style.css
    └── script.js
```

---

## Setup & Run

```bash
pip install -r requirements.txt

# 1. Provision a cloud MySQL instance (Aiven free tier) and set .env:
#    SCOUT_DB_HOST, SCOUT_DB_PORT, SCOUT_DB_USER, SCOUT_DB_PASSWORD,
#    SCOUT_DB_NAME, SCOUT_DB_SSL_CA=ca.pem

# 2. Create DB + tables, load data
python init_db.py
python load_static_data.py
python fetch_live_signals.py

# 3. Train models + generate inference-time category mappings
cd ml
python train_risk_classifier.py
python train_delay_forecast.py
python generate_category_mappings.py
cd ..

# 4. Or run the whole pipeline via DVC
dvc repro
dvc push

# 5. Run the agent pipeline locally
cd agents
python main.py <shipment_id>
python batch_run.py 30

# 6. Run the API locally
uvicorn api:app --reload

# 7. Serve the frontend locally
cd ../frontend_scout
python -m http.server 5500
```

**Production deployment:**
- **Backend**: Render web service, start command `cd agents && uvicorn api:app --host 0.0.0.0 --port $PORT`, env vars mirrored from `.env`, `ca.pem` committed to the repo.
- **Frontend**: static GitHub Pages deploy of `frontend_scout/`, pointing at the Render API URL.
- **Database**: Aiven MySQL (SSL-required), reachable from both local dev and Render.

---

## Challenges Faced & How They Were Solved

**1. Silent data loss from a broken aggregation (180K → 187 rows).**
An early `groupby().transform()` chain in the ingestion script misaligned indices and silently produced `NaN` for nearly every supplier's `lead_time_days`, which the feature pipeline then dropped. Fixed by replacing the transform chain with a clean `groupby().mean()` mapped back onto suppliers by region — and by making row-count sanity checks a standard step after every transform.

**2. Unlinked foreign keys (`shipments.supplier_id` always NULL).**
The original ingestion script never assigned `supplier_id` on insert. Fixed by reading back auto-generated `supplier_id`s after inserting suppliers and mapping them onto shipments via a natural key (region + category) before insert.

**3. MySQL password containing `@` breaking connection strings.**
Safe as a `pymysql.connect()` kwarg, but breaks a SQLAlchemy URI. Solved with `urllib.parse.quote_plus()` for URI-based connections only.

**4. Hyperparameter tuning plateau on the delay regressor.**
A 25-candidate randomized search barely moved R² (0.278 → 0.278). Diagnosed as a genuine signal ceiling — `scheduled_days` was collinear with `transport_mode`, and the target carries irreducible noise consistent with public benchmarks on this dataset — rather than continuing to tune blindly.

**5. Rapidly deprecating Gemini model identifiers.**
`gemini-1.5-flash` → `gemini-2.5-flash` → `gemini-3.6-flash`, each retired mid-build. Solved by treating the model name as a single config value and reading the API's own error message for the current replacement.

**6. Migrating to cloud MySQL broke relative file paths.**
Moving from local MySQL to Aiven required an SSL CA certificate, and its relative path (`ca.pem`) resolved differently depending on which directory a script was *run from* — breaking on Render, where the working directory (`agents/`) didn't match the certificate's location. Fixed by resolving the path relative to `db_connect.py`'s own file location (`os.path.dirname(os.path.abspath(__file__))`) instead of the process's current working directory, making it deployment-environment-agnostic.

**7. Production out-of-memory crash from reusing the training-time feature pipeline at inference.**
The API initially called the same `build_features()` used for training — reloading and reprocessing the **entire 180K-row dataset** just to score a single shipment. Combined with MLflow, LangChain, and pandas already loaded, this exceeded Render's 512MB free-tier limit and crashed the instance. Fixed by writing a separate, lightweight inference-time feature builder (`inference_features.py`) that computes a single shipment's features via targeted SQL `COUNT`/`AVG` queries instead of a full-table load — and by precomputing the categorical encoding scheme once via `generate_category_mappings.py` (lightweight `SELECT DISTINCT` queries) so inference never needs the full dataset in memory at all. This is a broader lesson: **training-time and inference-time feature computation are different problems and should not share the same code path** once memory or latency constraints apply.

**8. An overly broad `except ValueError` silently mislabeled real errors as "not found."**
After the memory fix, a genuine `ValueError` from XGBoost (a dtype mismatch — MySQL's `AVG()` returns `Decimal`, which pandas stores as `object`, not `float`) was being caught by the same handler used for "shipment not found," returning a misleading 404 instead of the real error. Fixed in two places: a dedicated `ShipmentNotFoundError` exception so only genuine not-found cases return 404, and explicit `float()` casting on all SQL aggregate results before feeding them to the model. The frontend was also hardcoding a generic "not found" message for any 404 response, which masked the real API error text during debugging — fixed by having the frontend display the API's actual `detail` field.

**9. Dependency conflicts and Python version drift breaking cloud builds.**
Render defaulted to Python 3.14 (no prebuilt `pyarrow` wheel yet, forcing a failing source build), and an unused `dagshub` package conflicted with `langgraph`'s `httpx` requirement. Fixed by pinning Python via `.python-version`, and by removing `dagshub` entirely since the code only ever talks to DagsHub through MLflow's own tracking URI + auth env vars — the package was dead weight.

---

## Results

| Model | Metric | Value |
|---|---|---|
| Risk classifier | Accuracy | 70.2% |
| Risk classifier | ROC-AUC | 0.757 |
| Risk classifier | Precision / Recall | 0.87 / 0.57 |
| Delay forecaster | MAE | 0.98 days |
| Delay forecaster | R² | 0.28 |

Both models are versioned in the MLflow model registry on DagsHub, with confusion matrix and feature-importance artifacts logged per run, and the full training pipeline reproducible via `dvc repro`.

---

## Conclusion & Key Takeaways

- **Debugging discipline mattered more than model choice.** The two biggest performance jumps (187 → 180,519 rows; 57.9% → 70.3% accuracy) came from fixing silent data-pipeline bugs, not from tuning or trying new algorithms.
- **Leak-free feature engineering is a discipline, not an afterthought.** Every "supplier history" feature was deliberately computed as an expanding window shifted by one, to prevent a shipment's own outcome from leaking into its own prediction.
- **Knowing when to stop tuning is itself a skill.** Recognizing a genuine data ceiling prevented burning further cycles chasing marginal gains that weren't there.
- **Training-time and production-time constraints are genuinely different problems.** A feature pipeline built for offline training (load everything, compute globally) can silently sink a production service under real-world memory limits — the fix wasn't "use a bigger server," it was recognizing that inference only ever needs *one row's worth* of context, computable with targeted queries instead of a full-dataset load.
- **Error handling shapes debuggability as much as correctness.** An overly broad exception handler and a frontend that hardcoded its own error message both actively hid the real failure for multiple debugging cycles — narrowing exception types and always surfacing the backend's real error text turned out to be as important as the underlying fix itself.
- **End-to-end ownership**: this project spans data engineering, cloud database migration, MLOps, ML modeling, agentic orchestration, backend deployment, and frontend delivery — the full stack a modern ML/AI engineering role expects, on a real, underserved business problem, deployed and publicly reachable rather than left as a notebook.

---

## Future Work

- Migrate `RandomizedSearchCV` to Bayesian optimization (Optuna) for more efficient search.
- Add SHAP-based explanations to replace the current feature-importance proxy.
- Add automated retraining triggers (drift monitoring) as new live risk events accumulate.
- Extend live-fetch to port-congestion and customs-delay data sources.
- Add authentication and rate-limiting to the public API before wider release.
- Restrict CORS to the deployed frontend origin instead of `*`.
