# 🗺️ Predict Jobs — Improvement Plan

**Created:** July 8, 2026  
**Status:** Planning  
**Priority Order:** Fix bugs first → Rebuild data layer → Improve UI

---

## Phase 1: Bug Fixes (Immediate)

### BUG-1 [HIGH] — Frontend double-filters by topClients
- **File:** `predict-jobs-dialog.component.ts` line 608
- **Issue:** Frontend checks if company is in `topClients` array AFTER backend already filtered by `company_cluster_profiles`. Since `topClients` only shows top 5 by volume, most companies show ZERO results.
- **Impact:** T-Mobile India shows 0 clusters. Any company with <1000 jobs is invisible.
- **Fix:** Remove the `topClients` local filter when company param was sent to backend API.

### BUG-2 [HIGH] — Company dropdown only shows 50 of 118 companies
- **File:** `predict-jobs-dialog.component.ts` `loadFilterOptions()`
- **Issue:** Dropdown populated from `topClients` arrays (only top clients by volume). 68 companies can't even be selected.
- **Impact:** Intel (8,910 jobs!), Nike (4,355), Motorola (3,134) — all invisible in dropdown.
- **Fix:** New API endpoint `GET /api/predictions/companies` → `SELECT DISTINCT company_name FROM company_cluster_profiles ORDER BY company_name`.

### BUG-3 [MEDIUM] — Summary KPIs ignore region & company filters
- **File:** `api/routes/predictions.py` `get_forecast_summary()`
- **Issue:** Summary endpoint has no filter params. Shows global 8,068 total even when user selects India + T-Mobile.
- **Impact:** Misleading KPI cards that don't match the filtered cluster list below.
- **Fix:** Add `region` and `company` query params, filter the aggregation SQL.

### BUG-4 [MEDIUM] — Skills endpoint ignores all filters
- **File:** `api/routes/predictions.py` `get_skill_demand()`
- **Issue:** No region/company/quarter parameters. Always returns global skills.
- **Impact:** User selects India but sees US-dominated global skill demand.
- **Fix:** Add filter params and apply to the skill aggregation query.

### BUG-5 [LOW] — Client names have country prefix ("IN Microsoft")
- **File:** `demand_forecasts.top_clients` data
- **Issue:** Client names stored as "IN PWC", "IN Microsoft" instead of clean names.
- **Fix:** Strip `XX ` prefix in the API response or clean during data ingestion.

---

## Phase 2: Data Quality (Sprint 1)

### DATA-1 — 39 companies missing from cluster profiles
- Intel Corp (8,910 jobs), Nike (4,355), Motorola (3,134), Kaiser (1,831), etc.
- **Fix:** Run profile generation script for all companies with ≥50 jobs.

### DATA-2 — Remove AI hallucination layer
- `job_forecast_roles` has 70 rows of GPT-invented numbers (no statistical basis).
- These show up on the dashboard as if they're real predictions.
- **Fix:** Drop `job_forecast` and `job_forecast_roles` tables from the prediction flow. Use ONLY `demand_forecasts` (the trained ML model output).

### DATA-3 — Status filter (from proposal Phase 1.2)
- Currently counting CANCELLED/EXPIRED/DRAFT jobs as "demand" — inflates numbers.
- **Fix:** Add status whitelist (FILLED, ACTIVE/OPEN only) to data preprocessing.

---

## Phase 3: Cluster Quality (Sprint 2)

### CLUSTER-1 — Split incoherent "General" clusters
Clusters that mix incompatible professions should be split:

| Current Cluster | Problem | Proposed Split |
|----------------|---------|----------------|
| US \| Specialist - General | Mixes IT, Supply Chain, HR, Admin | Split into: IT Specialist, Logistics Specialist, HR Specialist |
| US \| Operations - General | Mixes Data Center Ops, Business Ops, Field Ops | Split into: IT Operations, Business Operations |
| IN \| Operations - General | Mixes HR Ops (stable) with Sales Ops (spiky) | Split into: HR Operations, Sales Operations |
| US \| Associate - General | Mixes Scientists, Test, Marketing, QA | Split into: Lab Associate, Test Associate, Business Associate |
| US \| Developer | Mixes Network Specialist, Java, Game Dev, Web | Split into: Java Developer, Web Developer, Full Stack Developer |

### CLUSTER-2 — Merge too-volatile specific clusters
Clusters with MAPE >100% should be merged UP:

| Volatile Cluster | MAPE | Merge Into |
|-----------------|------|------------|
| US \| DevOps Engineer | 264% | US \| Engineering - General (or Cloud & Infrastructure) |
| US \| IT Analyst | 128% | US \| Business Analyst |
| US \| Design Engineer | 54% | US \| Engineering - General |

### CLUSTER-3 — Add sub-category breakdown (no re-clustering needed)
Instead of breaking the model, show predictions at cluster level WITH breakdown from actual data:

```
US | Developer — Forecast: 45 hires (Q3 2026)
├── Java Developers:      18 (40% based on last 4 quarters)
├── Full Stack:           12 (27%)
├── Web Developer:         8 (18%)
├── Mobile/Android:        4 (9%)
└── Other:                 3 (6%)
```

The ML predicts the total (which it's good at for high-volume clusters), and the split is derived from historical proportions. **Concrete, not hallucinated.**

---

## Phase 4: Prediction Engine Rebuild (Sprint 3)

### ENGINE-1 — Numbers ONLY from ML model
- ALL numeric predictions come from `demand_forecasts` (trained RF+LightGBM ensemble)
- NO GPT-generated counts anywhere
- Confidence intervals from MAPIE (mathematically guaranteed)

### ENGINE-2 — AI used ONLY for text summaries
- AI reads the ML model's numbers → writes explanation
- Example: "Software Engineering demand growing 18% in Q3, driven by cloud adoption. Recommend sourcing AWS-certified candidates."
- AI NEVER invents a number

### ENGINE-3 — Company-weighted predictions
- New endpoint: `GET /api/predictions/company/{name}`
- Shows: "T-Mobile represents ~5% of the Developer cluster demand in India"
- Prediction: "Expect 2-3 T-Mobile Developer openings in Q3" (cluster total × company share)

### ENGINE-4 — Skills derived from ACTUAL JD data
- For each cluster, aggregate skills from real `updated_job_records.skills` field
- NOT from GPT guessing what skills might be needed
- Show: "Based on 246 actual JDs for this role, top skills are: Java (89%), Spring Boot (67%), AWS (54%)"

### ENGINE-5 — Filter-aware KPIs
- When user selects Region + Company + Quarter:
  - Summary cards show filtered totals
  - Skills show filtered demand
  - Everything is consistent end-to-end

---

## Phase 5: UI Improvements (Sprint 4)

### UI-1 — Company autocomplete (replace dropdown)
- 118+ companies is too many for a dropdown
- Use autocomplete/search input
- Load from `company_cluster_profiles`

### UI-2 — Show data provenance
- Each prediction should show: "Based on X actual jobs in last Y months"
- Each skill should show: "Extracted from Z real JDs"
- No more mystery numbers

### UI-3 — Sub-category drilldown
- Click a cluster → see title-level breakdown with proportional allocation
- Real job titles, real companies, real historical patterns

### UI-4 — Confidence visualization
- Grade A/B: Solid bars with tight ranges
- Grade C: Dashed bars with wider ranges + "⚠️ Directional only"
- Grade D: Hidden by default, shown only if user unchecks "Reliable only"

---

## Success Criteria

After all phases:
- [ ] Filtering by any company + region + quarter shows correct, consistent results
- [ ] Every number shown can be traced back to either ML model or actual data (zero hallucination)
- [ ] KPI cards always reflect the active filters
- [ ] Sub-category breakdown gives concrete role-level answers
- [ ] Company-weighted view shows realistic per-client predictions
- [ ] All 118+ companies are searchable and show their clusters
- [ ] "IN PWC" displays as "PWC" 
- [ ] Volatile clusters (MAPE>100%) are hidden or merged

---

## Estimated Timeline

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 1 (Bug fixes) | 1 day | None |
| Phase 2 (Data quality) | 1 day | None |
| Phase 3 (Cluster quality) | 2-3 days | Phase 2 |
| Phase 4 (Engine rebuild) | 3-4 days | Phase 2 + 3 |
| Phase 5 (UI improvements) | 2-3 days | Phase 4 |
| **Total** | **~10-12 days** | |
