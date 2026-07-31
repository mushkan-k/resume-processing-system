# All Changes — Simple Explanation
> Updated: July 9, 2026

---

## What was wrong (before fixes)

| # | Problem | Impact |
|---|---------|--------|
| 1 | June 2026 data was missing from dashboard | Dashboard showed June as "predicted" even though it's already past |
| 2 | 98 clusters vanished when switching to Q3/Q4 | Q1 showed 145 clusters, Q3 showed only 49 — confused leadership |
| 3 | Company filter was FAKE | Selecting "Dell" showed 123 demand — but that was the FULL cluster, not Dell's share |
| 4 | KPI cards changed every time you clicked a quarter | No stable "year at a glance" — numbers kept jumping |
| 5 | Company dropdown was nearly empty | Only showed ~15 companies (from topClients field) instead of all 118 |
| 6 | No way to know if data was real or predicted | Dashboard showed numbers without saying "this is actual" vs "this is forecast" |

---

## What was fixed

### Fix 1: June 2026 actuals backfilled
**File:** `scripts/load_actuals.py`

**Before:** Dashboard queried `job_records` table which only had Jan–May data.

**After:** Now uses `updated_job_records` (the complete table with June data).

**Result:**
- Jan–Jun 2026 = **4,425 actual openings** (all marked `data_type = 'actual'`)
- Jul–Dec 2026 = predictions from ML model

---

### Fix 2: Missing 98 clusters now have forecasts
**File:** `scripts/backfill_missing_forecasts.py`

**Before:** Only 49 clusters had ML predictions. When you switched from Q2 to Q3, 96 clusters just disappeared.

**After:** All 147 clusters now have data in every quarter.

**How it works:**
- The 49 clusters with strong ML predictions → `is_reliable = TRUE`
- The 98 extra clusters → simple extrapolation from their actual history → `is_reliable = FALSE`
- "Reliable only" toggle hides the 98 low-confidence ones

**Think of it like:**
> You have 147 students. 49 took a proper exam (graded). 98 only did homework (estimated grade). You can show all 147 or filter to just the 49 with real exam scores.

---

### Fix 3: Company filter now shows PROPORTIONAL demand
**File:** `api/routes/predictions.py`

**Before:**
```
Select "Dell" → Shows "Operations - General = 123 demand"
(But 123 was the TOTAL for ALL 38 companies in that cluster!)
```

**After:**
```
Select "Dell" → Shows "Operations - General = 6 demand"
(Dell has 26 out of 611 openings = 4.25% share → 123 × 4.25% = 6)
```

**Simple analogy:**
> A building has 123 apartments. 38 companies rent there. Dell rents 6. Before, clicking Dell said "123 apartments" (the whole building). Now it correctly says "6 apartments" (Dell's share).

---

### Fix 4: KPI header is now CONSTANT (year-at-a-glance)
**File:** `predict-jobs-dialog.component.ts`

**Before:** Click Q1 → KPI shows "2,393". Click Q3 → KPI changes to "2,589". Confusing.

**After:** 
```
┌──────────────────────────────────────────────────────┐
│ 4,425          4,767           9,192         92.5%   │
│ Actual(Jan-Jun) Forecast(Jul-Dec) Full Year  Accuracy│
├──────────────────────────────────────────────────────┤
│  Q1: 2,393    Q2: 2,032    Q3: 2,589    Q4: 2,178  │
│  ████████     ████████     ▒▒▒▒▒▒▒▒     ▒▒▒▒▒▒▒▒  │
│  (actual)     (actual)     (predicted)   (predicted) │
└──────────────────────────────────────────────────────┘
```
This NEVER changes. Filters only affect the cluster list and skills section below.

**New API endpoint:** `GET /api/predictions/year-summary` — returns full year data always.

---

### Fix 5: Company dropdown now shows all 118 companies
**File:** `predict-jobs-dialog.component.ts`

**Before:** Dropdown pulled companies from `topClients` field (2-3 per cluster = ~15 unique).

**After:** Calls `GET /api/predictions/companies` endpoint which returns all 118 companies from the database.

---

### Fix 6: `dataType` field added to API responses
**File:** `api/routes/predictions.py`

Every data point now tells the frontend whether it's real or predicted:

```json
{
  "month": "2026-03",
  "demand": 67,
  "dataType": "actual"     ← GREEN styling, verified icon
}
{
  "month": "2026-08",
  "demand": 49,
  "dataType": "predicted"  ← PURPLE styling, dashed border
}
```

The frontend uses this to:
- Show green badge for actual quarters
- Show purple dashed badge for predicted quarters  
- Show a context message: "Showing real historical hiring data for Q1"

---

## Summary Table

| What | Before | After |
|------|--------|-------|
| June 2026 | ❌ Missing | ✅ 538 openings loaded |
| Q3/Q4 clusters | 49 | 147 (98 extrapolated) |
| Company filter: Dell | 123 (fake, full cluster) | 6 (real, proportional) |
| KPI header | Changes every filter click | Constant year-at-a-glance |
| Company dropdown | ~15 companies | 118 companies |
| Data type labels | None | actual / predicted / mixed |
| Employee classification | 91% | 94.8% (1,359/1,434) |

---

## Files Changed

| File | What was changed |
|------|-----------------|
| `scripts/load_actuals.py` | Uses `updated_job_records`, includes June, uses DB `role_cluster` column |
| `scripts/backfill_missing_forecasts.py` | NEW — generates naive forecasts for 98 missing clusters |
| `api/routes/predictions.py` | Proportional company demand, `dataType` field, year-summary endpoint |
| `predict-jobs-dialog.component.ts` | Year-at-a-glance header, company dropdown from API, dataType styling |

---

## Leadership Decisions Needed

See `docs/MASTER_PLAN.md` — two decisions pending:
1. **Openings vs Fills:** Should dashboard show demand (positions posted) or hires (positions filled) or both?
2. **Default view:** Show all 147 clusters or only the 49 reliable ones by default?
