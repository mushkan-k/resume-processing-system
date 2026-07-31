# Resume Processing System — Master Plan
> Updated: July 9, 2026

---

## 🎯 Leadership Decisions Pending

### 1. What should the dashboard forecast?
| Option | Metric | Jan–Jun Total |
|--------|--------|---:|
| A) Current | Openings (positions posted) | 4,436 |
| B) Add | Fills (people placed) | 615 (14% fill rate) |
| C) Both | Demand as primary, Fill Rate as secondary | |

### 2. Default view: Show all clusters or reliable only?
| Option | Clusters in Q3 | Trade-off |
|--------|:-:|---|
| A) Reliable only (current default) | 49 | Clean accuracy, but 98 clusters vanish |
| **B) Recommended: Show all** | 147 | Full picture with confidence badges |

---

## 🔴 NEXT — Drilldown Fixes (Data exists but not wired up)

### ISSUE-1: "No hiring history available"
**Table:** `cluster_hiring_history` — 27,738 rows (quarterly hires since 2017)  
**Root cause:** `/api/predictions/cluster/{name}` doesn't query this table  
**Fix:** Add query. Note: table uses `cluster_name = 'Developer'` (no prefix), API uses `'US | Developer'` → split prefix.

### ISSUE-2: "No job decomposition data"
**Table:** `cluster_job_decomposition` — 16,860 rows (top job titles per cluster)  
**Fix:** Query top 8 titles by jd_count for the cluster.

### ISSUE-3: "No external market data"
**Table:** `market_skill_signals` — 50 rows (LinkedIn/ThoughtWorks/JetBrains trends)  
**Fix:** Match cluster's role to `relevant_roles` JSON field.

### ISSUE-4: Skill Evolution empty
**Table:** `jd_skill_timeline` — 94,492 rows (skill frequency per quarter per cluster)  
**Fix:** Compare recent 2 quarters vs older 2 quarters → rising/declining/emerging.

### ISSUE-5: Actual vs Predicted quarters look IDENTICAL
| Element | Actual Quarter (Q1/Q2) | Predicted Quarter (Q3/Q4) |
|---------|----------------------|--------------------------|
| Border | 🟢 Green solid | 🟣 Purple dashed |
| Label | "Actual Hires" | "Forecasted Demand" |
| Accuracy | "Verified ✓" | "91% model accuracy" |
| Bars | Solid green | Purple + confidence range |
| Badge | 📊 ACTUAL DATA | 🔮 AI PREDICTION |

---

## 🏗️ Model Architecture

```
DATA: updated_job_records (190K+ records, 2017–Jun 2026)

FEATURES (22):
├── Macro: UNRATE, JOLTS, yield_spread, consumer_sentiment
├── Seasonal: Q1, Q2, Q3 dummies
├── Autoregressive: lag1, lag2, lag3, lag6, lag12
├── Rolling: 3m_mean, 6m_mean, 12m_mean
├── Leading: fills_lag1, fill_rate_lag1
├── Momentum: yoy_change, trend_slope, jolts_change
└── Identity: region_enc, role_enc, combo_avg_vol

MODELS:
├── Random Forest (800 trees, depth=14)
├── LightGBM (800 trees, lr=0.03, depth=10)
└── Ensemble = (RF + LGB) / 2

UNCERTAINTY: MAPIE conformal intervals (90% coverage)

VALIDATION:
├── Train: 2022-01 → 2025-12
├── Test: Q1 2026 holdout only
├── Portfolio MAPE: 7.6%
├── Median cluster MAPE: 18.9%
└── Interval coverage: 95.7%

COVERAGE: 49 ML clusters + 98 naive extrapolation
```

### Model Weaknesses
| # | Issue | Impact |
|---|-------|--------|
| 1 | Single holdout validation (Q1 only) | May overfit to one quarter |
| 2 | No Q2 data in training | 6 months unused |
| 3 | FRED signals stale (May 2026) | Economic context outdated |
| 4 | No retrain pipeline | Manual notebook only |
| 5 | 67% clusters are naive extrapolation | Low confidence majority |
| 6 | No skill-demand forecasting | Can't predict hot skills |

---

## 📋 Prioritized Work Remaining

### P2 — Drilldown Data Wiring (HIGH)
| # | Task | Est. |
|---|------|------|
| 1 | Wire `hiringHistory` in cluster detail API | 1h |
| 2 | Wire `jobDecomposition` in cluster detail API | 30m |
| 3 | Wire `marketContext` in cluster detail API | 30m |
| 4 | Wire `skillEvolution` from jd_skill_timeline | 1h |
| 5 | Different design for actual vs predicted | 2h |

### P3 — Model Improvements
| # | Task | Est. |
|---|------|------|
| 6 | Re-train with Q2 2026 data | 3h |
| 7 | Rolling cross-validation | 2h |
| 8 | Update FRED signals to Jul 2026 | 30m |
| 9 | Extend forecast to Q1 2027 | 1h |
| 10 | Expand viable clusters (lower threshold) | 2h |

### P4 — Automation
| # | Task | Est. |
|---|------|------|
| 11 | Create `scripts/retrain_model.py` | 4h |
| 12 | Monthly FRED data refresh | 1h |
| 13 | Expand market signals beyond Java | 2h |
| 14 | Recency-weighted company proportions | 2h |

### P5 — Frontend Polish
| # | Task | Est. |
|---|------|------|
| 15 | Separate loading signals (race condition) | 30m |
| 16 | Empty state when no clusters match | 30m |
| 17 | Skeleton/shimmer loading | 1h |

---

## 📊 Database — Unused Tables (need wiring)

| Table | Rows | Should Feed |
|-------|------|-------------|
| `cluster_hiring_history` | 27,738 | Drilldown → Quarterly Trend chart |
| `cluster_job_decomposition` | 16,860 | Drilldown → Top Roles Hired |
| `market_skill_signals` | 50 | Drilldown → Market Demand tab |
| `jd_skill_timeline` | 94,492 | Drilldown → Skill Evolution |

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `api/routes/predictions.py` | Prediction API (1,095 lines) |
| `predict-jobs-dialog.component.ts` | Dashboard UI (747 lines) |
| `prediction-role-drilldown.component.ts` | Drilldown UI (617 lines) |
| `scripts/load_actuals.py` | Backfill actuals |
| `scripts/backfill_missing_forecasts.py` | Naive forecasts for 98 clusters |
| `notebooks/07_enhanced_global.ipynb` | Model training |
| `docs/CHANGES_EXPLAINED.md` | All completed fixes documented |
