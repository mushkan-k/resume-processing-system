# Session Context — Resume Processing System + Predictions Dashboard

> **Last Updated:** July 21, 2026
> **Purpose:** Provide full continuity if starting a new chat session. Read this file first.

---

## 1. Project Overview

A staffing/consulting company system that:
1. **Pulls JDs from JobDiva API** → classifies them into role clusters → stores in MySQL
2. **Forecasts future demand** using a Bayesian model (HierBayes with NumpyRo/JAX)
3. **Displays predictions** on an Angular dashboard for CEO presentation (first week of August 2026)
4. **Matches resumes** to JDs using skill extraction

---

## 2. Architecture

| Layer | Tech | Port | Location |
|-------|------|------|----------|
| Backend API | FastAPI + Python 3.13 | 8080 | `c:\Users\serveradmin\Desktop\resume-processing-system\` |
| Frontend | Angular 19 (standalone components, signals) | 4200 | `c:\Users\serveradmin\Desktop\resumeGrid_2.0\Resume_Grid_App\Resume_Grid_App\` |
| Database | MySQL 8 (Docker container `resume_mysql`) | 3305 | DB: `resume_processing`, User: `resume_user`, Pass: `resume_password` |
| Model | NumpyRo/JAX Bayesian inference | — | `scripts/generate_hier_forecasts.py` |

### How to Run
```powershell
# Backend (from resume-processing-system/)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

# Frontend (from Resume_Grid_App/Resume_Grid_App/)
npx ng serve --proxy-config proxy.conf.json
```

---

## 3. Key Files

### Backend
| File | Purpose |
|------|---------|
| `api/routes/predictions.py` | All prediction endpoints (`/clusters`, `/cluster/{name}`, `/summary`, etc.) |
| `api/routes/jds.py` | JD listing/detail (reads from `job_descriptions` table synced from JobDiva) |
| `api/routes/skills.py` | Skill rankings, practice rankings |
| `api/routes/employee.py` | Employee/resume data |
| `api/main.py` | FastAPI app, background JD sync task |
| `scripts/generate_hier_forecasts.py` | The forecasting model — generates `demand_forecasts` rows |

### Frontend
| File | Purpose |
|------|---------|
| `src/app/features/jd-catalog/predictions-page.component.ts` | Main predictions dashboard page |
| `src/app/features/jd-catalog/prediction-role-drilldown.component.ts` | Cluster detail modal (hiring history, job decomposition, skills, forecast chart) |
| `src/app/features/jd-catalog/predict-jobs-dialog.component.ts` | Shared interfaces (`PredictionSummary`, `PredictionCluster`, etc.) |
| `src/app/features/jd-catalog/jd-catalog.component.ts` | JD catalog page |

### Data
| File | Purpose |
|------|---------|
| `data/clean_42k_v1.pkl` | 46,590 classified JD rows (training data — enriched from original 42k) |
| `data/clean_42k_v1_backup.pkl` | Original 42k backup before enrichment |
| `data/title_to_cluster.pkl` | Title → cluster mapping v2 (12,390 titles → 43 clusters) |
| `data/title_to_cluster_v1_backup.pkl` | Original v1 mapping backup |
| `data/title_to_cluster_v2.csv` | CSV review copy of v2 mapping |
| `data/demand_forecasts.json` | Exported forecasts (also stored in MySQL) |
| `data/market_signals/bls_oes_data.json` | BLS employment/wage data for 22 SOC codes |
| `data/market_signals/onet_skills_taxonomy.json` | O*NET skills for 29 clusters (4,322 tech skills) |
| `data/market_signals/career_pages/` | 19 company career page scrapes (898 total jobs) |

### Scripts (new this session)
| File | Purpose |
|------|---------|
| `scripts/recluster_v2.py` | Re-clustering pipeline (TF-IDF + K-Means k=55, manual overrides) |
| `scripts/reclassify_db.py` | Updates role_cluster column in DB with new mapping |
| `scripts/augment_training_data.py` | Adds Gap Inc + career page gap-fills to training data |
| `scripts/fix_fills.py` | Fixes fills column using job_status from raw DB (19,688 corrections) |
| `scripts/load_actuals.py` | Loads Q1/Q2 actuals into demand_forecasts with "Region \| Role" format |
| `scripts/jobdiva_jd_resume_pipeline.py` | Full JD→Resume extraction pipeline from JobDiva API |
| `scripts/fetch_missing_title_skills.py` | Fetches JDs from JobDiva for titles missing skills in decomposition (39/40 fetched) |
| `scripts/fix_garbage_skills.py` | Fixed 132/138 "Over X Year S" garbage entries via JD re-extraction |
| `scripts/fix_garbage_skills_v2.py` | Found 8 remaining garbage entries (skipped — no JD available) |
| `scripts/fix_garbage_skills_v3.py` | Fixed last 8 garbage entries using title-based LLM inference |
| `scripts/predict_skills.py` | **Skills Forecasting Model v2** — predicts skills per cluster for Q3/Q4 2026 |

---

## 4. Database Tables (MySQL)

| Table | Purpose |
|-------|---------|
| `demand_forecasts` | Core table — has `data_type` ('actual' or 'predicted'), `cluster_name`, `forecast_month`, `demand_predicted`, `mape`, `mase`, `is_reliable`, `top_skills`, `top_locations`, `top_clients` |
| `cluster_hiring_history` | 12 quarters of historical hiring per cluster |
| `cluster_job_decomposition` | Job title breakdown within each cluster |
| `company_cluster_profiles` | Per-company skill/location profiles for clusters |
| `jd_extracted_skills` | NLP-extracted skills from actual JD text (2031 rows, 86 clusters, quarters 2024Q4–2026Q3) |
| `predicted_cluster_skills` | **Skills forecasting model output** — predicted skills per cluster per future quarter (172 rows, model v2) |
| `skill_trends` | Skill trend analysis (75,626 rows) — trending_score, quarters_active, first/last seen |
| `market_skill_signals` | External market signals for skills |
| `job_descriptions` | Raw JDs synced from JobDiva |
| `job_records` | Raw job records |
| `updated_job_records` | Updated/recent job records used for actual demand |
| `model_metrics` | Forecast model performance metrics |
| `employee` | Employee records |
| `resume` | Resume records linked to employees |
| `employee_skillset` | Extracted skills per employee |
| `skill_practice_mapping` | Maps skills to practices |
| `skill_ranking_history` | Historical skill rankings |
| `cluster_division_map` | Maps clusters to business divisions |

---

## 5. Data Flow for Predictions

```
JobDiva API (NewUpdatedJobRecords) 
  → job_descriptions table 
  → classify into clusters (title_to_cluster.pkl)
  → demand_forecasts (data_type='actual')

generate_hier_forecasts.py
  → Reads actuals from demand_forecasts
  → Runs NumpyRo Bayesian model (2 chains × 2000 samples)
  → Writes forecasts to demand_forecasts (data_type='predicted')
  → Also populates: cluster_hiring_history, cluster_job_decomposition, company_cluster_profiles
```

---

## 6. Model Details

- **Model:** HierBayes_v1_JOLTS — hierarchical Bayesian with JOLTS labor market data
- **Seasonal adjustment:** Monthly seasonal indices derived from historical data per cluster
- **Forecast horizon:** 6 months (Jul–Dec 2026)
- **Performance:** MAPE 47.0%, coverage ~82%
- **Result:** 330 predicted rows, 55 region|role clusters, total 4,559 openings H2 2026 (1.02× H1 actuals)
- **Training data:** 46,590 records, 157 companies, fills corrected via job_status
- **Output format:** "Region | Role" (e.g., "US | Developer") — matches actuals format
- **TFT experiment:** Tested and rejected (35.9% MAPE but 16% inflated totals, poor coverage)

---

## 7. Predictions API Endpoints

### GET /api/predictions/clusters
Query params: `quarter`, `region`, `company`, `tier`
Returns: `{ total, quarters: [...], clusters: [...] }`

### GET /api/predictions/cluster/{name}
Returns: Full detail — hiring history (12 qtrs), job decomposition, skills, locations, clients, forecast months, grade, confidence

### GET /api/predictions/summary
Returns: High-level summary stats

---

## 8. Grading / Reliability System

**Thresholds (based on MAPE):**
| MAPE | Label | CSS Class |
|------|-------|-----------|
| ≤ 35% | Very Stable | `grade--very-stable` / `grade-very-stable` |
| ≤ 50% | Stable | `grade--stable` / `grade-stable` |
| ≤ 70% | Moderate | `grade--moderate` / `grade-moderate` |
| > 70% | Volatile | `grade--volatile` / `grade-volatile` |

**Tier system:**
- Tier 1 = `is_reliable = true` (model ran successfully for this cluster)
- Tier 2 = `is_reliable = false` (estimated/extrapolated)

**Confidence mapping:**
- No forecasts + has estimates → "estimate"
- No forecasts → "actual"
- Very Stable / Stable → "high"
- Moderate → "medium"
- Volatile → "low"

---

## 9. Filters Working

| Filter | Status | Notes |
|--------|--------|-------|
| Quarter | ✅ | Filters `forecast_month` in SQL |
| Region | ✅ | Mapped from `COUNTRY_TO_REGION` dict; plain cluster names get region from actuals cache |
| Company | ✅ | Filters via `company_cluster_profiles`; also matches by plain role name for predicted clusters |
| Tier | ✅ | Client-side: Tier 1 = `reliable`, Tier 2 = `!reliable` |

---

## 10. Cluster Deduplication Logic

When same role appears as both actual (e.g., "US | Developer") and predicted ("Developer"):
- Merges into one cluster
- Prefers the forecasted version's data
- Combines demand, skills, locations
- Result: 187 → 121 unique clusters

---

## 11. UI Features (predictions-page.component.ts)

- **KPI cards:** Total Demand, Reliable Clusters count, Avg Accuracy
- **Quarter bar:** Shows Q3/Q4 2026 with demand totals; context-aware (all quarters vs drilled quarter)
- **Filter pills:** Region dropdown, Tier dropdown, Company dropdown
- **Cluster cards:** Role name, tier badge, demand count, grade badge, top skills chips
- **Click card → opens drilldown modal** with full detail

---

## 12. Drilldown Modal (prediction-role-drilldown.component.ts)

- **KPI strip:** Demand, Accuracy %, Actual count, Forecast count, Reliability label
- **3 tabs:** Overview, Skills & Market, Forecast
- **Overview tab:** Hiring history bar chart (12 qtrs), job title decomposition bars, locations chips, top clients chips
- **Prediction basis footer:** "Model: HierBayes_v1_JOLTS | X months actuals + Y months forecast"

---

## 13. What Was Completed This Session

1. ✅ Seasonal variation in forecasts (Q3 ≠ Q4)
2. ✅ Region filter working (from actuals cache)
3. ✅ Cluster deduplication (actual + predicted merge)
4. ✅ Company filter working with predicted clusters
5. ✅ Hiring history populated (12 quarters)
6. ✅ Job decomposition populated
7. ✅ Locations/clients fallback from actuals
8. ✅ Grade labels changed: A/B/C/D → Very Stable/Stable/Moderate/Volatile
9. ✅ Grade thresholds relaxed (20/35/50 → 35/50/70) for better distribution
10. ✅ Frontend CSS updated for new grade labels
11. ✅ Drilldown shows grade label directly (removed "Grade" prefix)
12. ✅ Repo cleaned (logs, PNGs, temp files removed)
13. ✅ Re-clustered titles (12,390 → 43 clusters via TF-IDF + K-Means k=55)
14. ✅ Collected market data (BLS OES, O*NET skills, 19 company career pages)
15. ✅ Fixed Region|Role format mismatch (predictions now use "US | Developer" like actuals)
16. ✅ Enriched training data: career page gap-fills (+967 records), removed <5 opening companies (-189)
17. ✅ Fixed fills column using job_status from DB (19,688 corrections, zero fills 81.1% → 39.9%)
18. ✅ Re-ran Bayesian forecasts (330 rows, 55 clusters, MAPE 47.0%, total 4,559)
19. ✅ TFT experiment — tested twice, rejected (inflated totals, lower coverage vs Bayesian)
20. ✅ Built JobDiva JD+Resume pipeline script (`scripts/jobdiva_jd_resume_pipeline.py`) for senior
21. ✅ Fixed JD count accuracy (`source_type` mismatch: normalized 52 rows `full_text` → `jd_full_text`)
22. ✅ Per-title skill enrichment (39 + 4 targeted titles fetched from JobDiva, fuzzy matching for variants)
23. ✅ Garbage skills fixed — 140 total "Over X Year S" entries across all clusters
24. ✅ Healthcare/Clinical 500 error fixed (`UnboundLocalError: role` → `_plain_role`)
25. ✅ Title variant grouping in decomposition (e.g. "Structures III D/C/A/B" → single row)
26. ✅ Q1/Q2 actual quarters: proper per-title skills from real JDs with quarter filtering
27. ✅ Skills Forecasting Model v2 built and run (172 predictions, 86 clusters × Q3/Q4)
28. ✅ API updated: predicted quarters now pull from `predicted_cluster_skills` table
29. 🔄 API restart needed to verify predicted quarter skills display end-to-end

---

## 14. Skills Forecasting Model (`scripts/predict_skills.py`)

**Approach:**
1. Loads ALL `jd_extracted_skills` across 7 quarters (2024Q4–2026Q3, 2031 records, 86 clusters)
2. Also uses `skill_trends` table (75K rows) as supplementary signal
3. Classifies each skill per cluster:
   - **Foundation** (appears in 60%+ of quarters, or all skills if only 1 quarter available)
   - **Trending Up** (frequency increasing in recent half vs early half, or trend_score > 0.3)
   - **Emerging** (only in most recent quarter)
   - **Trending Down** (frequency decreasing — excluded from predictions)
4. Generates predicted skills per cluster for Q3/Q4 2026:
   - Foundation (top 10) + Trending Up (top 5) + Emerging (top 3) + Stable fill
   - Max 12 skills per cluster
5. Stores in `predicted_cluster_skills` table with confidence level

**Sample Results:**
- US | Software Engineer: Python, Kubernetes, Java, Automation, SQL, .NET, Terraform, CI/CD
- IN | Software Engineer: Java, AWS, Azure, Cassandra, Python, PostgreSQL, Generative AI
- IN | Engineering - General: Python, FAISS, LangGraph, Milvus, AutoGen, observability frameworks
- CO | Engineering - General: Python, Apache Airflow, Apache Spark, Delta Lake, NoSQL, SQL

**DB Table:** `predicted_cluster_skills` (cluster_name, quarter, predicted_skills JSON, skill_categories JSON, confidence, model_version)

---

## 15. Predicted Quarter Handling in API

In `api/routes/predictions.py`, the `/cluster/{name}` endpoint:
- Detects if the requested quarter is a **predicted quarter** (no actual `updated_job_records` exist for that date range)
- If predicted: queries `predicted_cluster_skills` for that cluster+quarter → overrides `topSkills` with forecasted skills
- Sets `skillsSource` to indicate skills are from the forecasting model
- No `jobDecomposition` shown for predicted quarters (no actual JDs to decompose)

---

## 16. Known Issues / Next Steps

| Issue | Notes |
|-------|-------|
| API needs restart + test | Verify that Q3/Q4 predicted quarters show forecasted skills from `predicted_cluster_skills` |
| Frontend drilldown for predicted quarters | Should show predicted skills clearly with "forecasted" label; no job decomposition for predicted quarters |
| Company filter shows full cluster demand | Dashboard shows total cluster demand when filtering by company — should show company's proportional share |
| Rebuild company_cluster_profiles | Needed for new 43 clusters to work with company filter properly |
| MAPE could improve | 47% — consider more market signals, longer history, or ensemble methods |
| Some clusters have few skills | Clusters with only 1 quarter of data have limited skill predictions |
| CEO presentation | First week of August 2026 — UI needs polish |

---

## 17. Environment Variables (.env)

```
DB_HOST=localhost
DB_PORT=3305
DB_USER=resume_user
DB_PASSWORD=resume_password
DB_NAME=resume_processing
JOBDIVA_CLIENT_ID=...
JOBDIVA_USERNAME=...
JOBDIVA_PASSWORD=...
```

---

## 18. Useful Commands

```powershell
# Check API
Invoke-RestMethod -Uri "http://localhost:8080/api/predictions/clusters" | ConvertTo-Json -Depth 3

# Check specific company
Invoke-RestMethod -Uri "http://localhost:8080/api/predictions/clusters?company=T-Mobile"

# Build Angular
cd C:\Users\serveradmin\Desktop\resumeGrid_2.0\Resume_Grid_App\Resume_Grid_App
npx ng build --configuration=development

# Run forecasts
cd C:\Users\serveradmin\Desktop\resume-processing-system\scripts
python generate_hier_forecasts.py

# MySQL access
docker exec -it resume_mysql mysql -u resume_user -presume_password resume_processing
```

---

## 19. Important Patterns

- **Cluster naming:** Both actuals and predictions now use "Region | Role" format (e.g., "US | Developer")
- **COUNTRY_TO_REGION mapping:** US→US, IN→India, MX/BR/CO/AR→LATAM, everything else→Other
- **Angular proxy:** `proxy.conf.json` routes `/api/*` to `http://localhost:8080`
- **All components are standalone** (Angular 19 pattern, no NgModules)
- **Signals used** for reactive state in Angular components
