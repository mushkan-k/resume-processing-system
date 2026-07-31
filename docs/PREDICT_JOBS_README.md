# 📊 Predict Jobs — How It Works

## Overview

The **Predict Jobs** feature forecasts future staffing demand — telling leadership *"Expect ~145 Software Engineer openings in Q3 2026, likely between 125–165"*. It combines two approaches:

1. **ML Forecasting Engine** — A trained Random Forest + LightGBM ensemble model that predicts monthly demand per role/region (the accurate one)
2. **AI-Powered Predictions** — Azure OpenAI (GPT-4o-mini) that analyzes quarterly trends and generates human-readable predictions with requirement profiles

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA SOURCES                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [Internal Job DB]     [FRED Economic Data]     [JD Full Text]   │
│   42,000+ records       4 macro signals          NLP-extracted    │
│   2022 – May 2026      Updated monthly           skills           │
│                                                                   │
└────────┬───────────────────────┬────────────────────┬────────────┘
         │                       │                    │
         ▼                       ▼                    ▼
┌─────────────────┐   ┌──────────────────┐   ┌───────────────────┐
│ Role Clustering  │   │ Feature Engine   │   │ Skills Extraction  │
│ 1000+ titles →   │   │ 23 features per  │   │ Boolean parsing +  │
│ 50 clusters      │   │ combo/month      │   │ NLP full-text      │
└────────┬─────────┘   └────────┬─────────┘   └────────┬──────────┘
         │                       │                      │
         ▼                       ▼                      │
┌─────────────────────────────────────────┐             │
│        ML ENSEMBLE MODEL                 │             │
│  Random Forest (800 trees) + LightGBM   │             │
│  + MAPIE Conformal Prediction Intervals  │             │
│  → Monthly demand per role/region        │             │
└────────────────────┬────────────────────┘             │
                     │                                   │
                     ▼                                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                     PREDICTION API                                │
│                                                                   │
│  GET /api/predictions       → Dashboard overview (quarterly)     │
│  GET /api/predictions/clusters → Cluster list with grades        │
│  GET /api/predictions/summary  → Portfolio accuracy + stats      │
│  GET /api/predictions/skills   → Skills ranked by demand         │
│  GET /api/predictions/quarterly → Quarter-over-quarter view      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Used

### 1. Internal Job Records (Primary Source)
- **Table:** `updated_job_records`
- **Records:** 42,000+
- **Time Range:** 2022 – May 2026 (4+ years)
- **Fields:** `title`, `company_name`, `division_name`, `city`, `state`, `country`, `issue_date`, `job_status`, `bill_rate_min`, `bill_rate_max`, `skills`, `openings`

### 2. FRED Economic Indicators (External Signals)

| Signal | Source | What It Tells Us |
|--------|--------|-----------------|
| **UNRATE** | Bureau of Labor Statistics | Unemployment rate — labor market tightness |
| **JTSJOL** | JOLTS Survey | Total job openings in the US — leading demand indicator |
| **T10Y2Y** | US Treasury | 10Y–2Y yield spread — recession probability signal |
| **UMCSENT** | Univ. of Michigan | Consumer sentiment — business confidence proxy |

These are stored in `/data/UNRATE.csv`, `/data/JTSJOL.csv`, `/data/T10Y2Y.csv`, `/data/UMCSENT.csv`.

### 3. Cleaned Job Dataset
- **File:** `/data/clean_42k_v1.pkl`
- Pre-processed dataset with normalized titles, region assignments, parsed fields

### 4. Role Clusters
- **File:** `/data/role_clusters.json`, `/data/title_to_cluster.pkl`
- Maps 1,000+ raw job titles → 50 standardized role clusters  
  (e.g., "Sr. Java Developer", "Java Dev II" → **"Software Engineer"**)

---

## Model Details

### Generation History

| Gen | Model | Approach | Accuracy |
|-----|-------|----------|----------|
| 1 | SARIMAX | One model per combo, quarterly | 58% (41.8% MAPE) ❌ |
| 2 | Random Forest | Global model, quarterly | 77% (22.9% MAPE) ✓ |
| **3** | **RF + LightGBM Ensemble** | **Global, monthly, FRED signals** | **92.4%** (18.9% MAPE) ✓✓ |

### Current Model (Generation 3)
- **Algorithm:** Ensemble of Random Forest (800 trees) + LightGBM
- **Training Data:** 2,456 monthly observations across 49 region/role combos
- **Features:** 23 engineered features per prediction (see below)
- **Intervals:** MAPIE (conformal prediction) for 90% confidence ranges
- **Granularity:** Monthly, aggregated to quarterly on the dashboard

### Feature Engineering (23 Features)

| Category | Features |
|----------|----------|
| **Autoregressive** | Last 1, 2, 3, 6, 12 month openings; 3/6/12-month rolling avg; last month fills + fill rate |
| **Momentum** | Year-over-year change; 6-month trend slope; JOLTS momentum |
| **Seasonality** | Quarter indicators (Q1, Q2, Q3 — Q4 is reference) |
| **Identity** | Region (encoded); role cluster (encoded); historical average volume |
| **Macro** | UNRATE, JTSJOL, T10Y2Y, UMCSENT (current values) |

### Prediction Unit
Each prediction is for a **Region | Role Cluster** combo:
- e.g., `"US | Software Engineer"`, `"IN | Operations - General"`, `"US | Data Engineer"`
- Total: **49 viable combos** (filtered from ~200+ by requiring ≥10 records/quarter)

---

## AI-Powered Layer (Azure OpenAI)

On top of the ML model, an **AI summary layer** provides:
- Human-readable trend explanations for each role
- Required skills per predicted role (what recruiters should look for)
- Expected companies and locations
- Actionable advice: *"Start sourcing Senior Java Developers with AWS experience now — demand growing 23%"*

### Configuration
- **Endpoint:** Azure OpenAI (`aiopsproject.openai.azure.com`)
- **Deployment:** `gpt-4o-mini`
- **Temperature:** 0.3 (focused, consistent predictions)
- **Prompt:** Receives quarterly historical totals + per-role profiles → outputs structured JSON

---

## Grading System

Each cluster gets a reliability grade:

| Grade | Criteria | Meaning | Dashboard Display |
|-------|----------|---------|-------------------|
| **A** | MAPE ≤ 20% & MASE < 1 | Plan confidently | High confidence (green) |
| **B** | MAPE ≤ 35% & MASE < 1 | Reliable for planning | High confidence (blue) |
| **C** | MAPE ≤ 50% | Directional signal | Medium confidence (orange) |
| **D** | MAPE > 50% | Low confidence | Low confidence (red) |

- **33 of 49 clusters are Grade A/B** (68% of total demand)
- Grade C/D clusters are flagged with warnings on the dashboard

---

## Validation

### How We Know It Works

| Method | Result |
|--------|--------|
| Holdout validation (Q1 2026) | **92.4% portfolio accuracy** |
| Prediction interval coverage | **95.7%** (actuals fall in range 96% of time) |
| Multi-quarter backtest | 78–97% across 4 quarters |

### Q4 Seasonal Correction
- Historical Q4/Q3 ratio: **0.86** (Q4 is always ~15% lower — holiday slowdown)
- Applied per-combo correction factor to Q4 forecasts
- Capped at 40% max adjustment

---

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/generate_forecast.py` | Main forecast: aggregates data → calls AI → stores predictions with requirement profiles |
| `scripts/generate_predictions.py` | Quarter predictions: Q1/Q2 actuals + Q3/Q4 AI predictions (simpler model) |
| `scripts/sync_predictions.py` | Sync prediction data to frontend static files |
| `scripts/patch_predictions.py` | Hotfix script for patching specific prediction values |
| `scripts/create_forecast_tables.py` | Creates DB schema for forecast tables |
| `notebooks/05_forecasting.ipynb` | Gen 1 — SARIMAX exploration |
| `notebooks/06_global_model.ipynb` | Gen 2 — Random Forest global model |
| `notebooks/07_enhanced_global.ipynb` | Gen 3 — Final ensemble + FRED + MAPIE |

---

## Database Tables

| Table | Purpose |
|-------|---------|
| `demand_forecasts` | Monthly ML predictions per cluster (49 combos × 6 months) |
| `quarter_predictions` | AI-generated quarterly summaries (titles, companies, practices) |
| `job_forecast` | High-level quarter totals (actual Q1/Q2, predicted Q3/Q4) |
| `job_forecast_roles` | Per-role predictions with skills, locations, companies |
| `jd_extracted_skills` | NLP-extracted skills from JD full text (used for "top skills" display) |
| `company_cluster_profiles` | Which companies historically hire which clusters |

---

## API Endpoints

| Endpoint | Returns |
|----------|---------|
| `GET /api/predictions` | Full dashboard data — clusters with quarterly demand, grades, ranges |
| `GET /api/predictions?region=India` | Filter by region |
| `GET /api/predictions?company=Wabtec` | Filter by company |
| `GET /api/predictions/clusters` | All clusters with accuracy grades |
| `GET /api/predictions/summary` | Portfolio accuracy, reliable count, total demand |
| `GET /api/predictions/skills` | Skills ranked by predicted demand volume |
| `GET /api/predictions/quarterly` | Quarter-over-quarter view (Q1-Q4 2026) |
| `GET /api/predictions/roles` | Per-role detail with requirement profiles |

---

## How to Re-run Predictions

```bash
# 1. Generate the ML-based monthly forecasts (from notebooks/07)
python scripts/generate_forecast.py

# 2. Generate AI quarterly predictions
python scripts/generate_predictions.py

# 3. Sync to frontend
python scripts/sync_predictions.py
```

---

## Key Metrics

| Metric | Value |
|--------|-------|
| Overall Accuracy | **92.4%** |
| Prediction Interval Coverage | **95.7%** |
| Roles Covered | **49** region/role combos |
| Forecast Window | Jul–Dec 2026 (Q3 + Q4) |
| Reliable Clusters (A/B) | **33 of 49** |
| Demand Coverage (reliable) | **68%** of total hiring |
| Features Per Prediction | **23** |
| Training Observations | **2,456** monthly data points |
| Training Time | < 2 minutes |
| Cost | $0 (FRED data is free, model runs locally) |

---

## Limitations

- 16 clusters are Grade C/D (insufficient data or too volatile)
- No client-level forecasting yet (predicts role+region, not per-client)
- Macro signals lag 1-2 months (Fed data release delay)
- Black swan events (like Q4 2025's −34% market drop) can't be predicted in advance
- AI summary layer depends on Azure OpenAI availability

---

## Next Steps (Potential)

- [ ] Client-level features for per-account predictions
- [ ] Internal pipeline data (recruiter activity, interview volume)
- [ ] Weekly granularity for short-term forecasting
- [ ] Auto-retrain monthly as new data arrives
- [ ] Alert system when actual demand deviates >20% from forecast
