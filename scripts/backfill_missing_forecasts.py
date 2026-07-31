"""
P0 #2 — Generate naive forecasts for missing clusters
=====================================================
98 clusters have actual demand (Jan-Jun 2026) but NO predictions for Q3/Q4.
They vanish from the dashboard when switching to future quarters.

Strategy:
- Use each cluster's average monthly demand as the baseline forecast
- Apply seasonal dampening for Q4 (Oct-Dec typically ~15% lower)
- Mark as is_reliable = False, model_used = 'Naive_Extrapolation'
- Confidence intervals: ±40% of point estimate (wide, reflecting uncertainty)

This gives the dashboard something reasonable to show without misleading anyone.
"""
import mysql.connector
import numpy as np

DB_CONFIG = {
    "host": "localhost",
    "port": 3305,
    "database": "resume_processing",
    "user": "resume_user",
    "password": "resume_password",
}

# Seasonal dampening factors (based on typical IT staffing seasonality)
# Q3 is usually strong, Q4 dips due to budget freezes / holidays
MONTH_FACTORS = {
    7: 1.0,   # Jul — strong
    8: 0.95,  # Aug — slight dip
    9: 0.90,  # Sep — budget reviews
    10: 0.85, # Oct — Q4 slowdown
    11: 0.75, # Nov — holidays approaching
    12: 0.65, # Dec — lowest hiring month
}

print("=" * 70)
print("P0 #2 — GENERATE FORECASTS FOR MISSING CLUSTERS")
print("=" * 70)

conn = mysql.connector.connect(**DB_CONFIG)
cur = conn.cursor(dictionary=True)

# ── Step 1: Find clusters with actuals but no predictions ──
cur.execute("SELECT DISTINCT cluster_name FROM demand_forecasts WHERE data_type = 'actual'")
actual_clusters = {r['cluster_name'] for r in cur.fetchall()}

cur.execute("SELECT DISTINCT cluster_name FROM demand_forecasts WHERE data_type = 'predicted'")
predicted_clusters = {r['cluster_name'] for r in cur.fetchall()}

missing = actual_clusters - predicted_clusters
print(f"\n  Clusters with actuals only (no predictions): {len(missing)}")

if not missing:
    print("  Nothing to do — all clusters have predictions!")
    cur.close()
    conn.close()
    exit()

# ── Step 2: Get actual monthly data for missing clusters ──
cur.execute("""
    SELECT cluster_name, forecast_month, demand_predicted,
           top_skills, top_locations, top_clients
    FROM demand_forecasts
    WHERE data_type = 'actual' AND cluster_name IN ({})
    ORDER BY cluster_name, forecast_month
""".format(','.join(['%s'] * len(missing))), list(missing))

cluster_data = {}
for row in cur.fetchall():
    cn = row['cluster_name']
    if cn not in cluster_data:
        cluster_data[cn] = {
            'months': [],
            'demands': [],
            'top_skills': row['top_skills'],
            'top_locations': row['top_locations'],
            'top_clients': row['top_clients'],
        }
    cluster_data[cn]['months'].append(row['forecast_month'])
    cluster_data[cn]['demands'].append(int(row['demand_predicted']))

# ── Step 3: Generate forecasts for Jul-Dec 2026 ──
insert_sql = """
    INSERT INTO demand_forecasts 
    (cluster_name, forecast_month, demand_predicted, demand_lower, demand_upper,
     model_used, mape, mae, mase, is_reliable, top_skills, top_locations, top_clients, data_type)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'predicted')
    ON DUPLICATE KEY UPDATE
        demand_predicted = VALUES(demand_predicted),
        demand_lower = VALUES(demand_lower),
        demand_upper = VALUES(demand_upper),
        model_used = VALUES(model_used),
        is_reliable = VALUES(is_reliable)
"""

MODEL_NAME = "Naive_Extrapolation"
inserted = 0
clusters_forecasted = 0

for cluster_name, data in cluster_data.items():
    demands = data['demands']
    n_months = len(demands)
    
    # Calculate baseline: weighted average (recent months weighted more)
    if n_months >= 3:
        # Give more weight to recent months
        weights = np.array([1.0 + 0.5 * i for i in range(n_months)])
        avg_demand = np.average(demands, weights=weights)
    else:
        avg_demand = np.mean(demands)
    
    # Skip clusters with negligible demand (< 0.5 avg)
    if avg_demand < 0.5:
        continue
    
    clusters_forecasted += 1
    
    # Calculate variability for confidence intervals
    if n_months >= 2:
        cv = np.std(demands) / max(np.mean(demands), 1)  # coefficient of variation
    else:
        cv = 0.5  # high uncertainty for single-month data
    
    # Estimate MAPE (higher for low-volume, volatile clusters)
    estimated_mape = min(30 + cv * 30, 80)  # Range: 30-80%
    
    for month_num in range(7, 13):
        forecast_month = f"2026-{month_num:02d}"
        seasonal_factor = MONTH_FACTORS[month_num]
        
        # Point estimate
        point = max(1, round(avg_demand * seasonal_factor))
        
        # Confidence interval (wider for less data)
        margin = max(1, round(point * (0.4 + cv * 0.2)))  # 40-60% margin
        lower = max(0, point - margin)
        upper = point + margin
        
        cur.execute(insert_sql, (
            cluster_name,
            forecast_month,
            point,
            lower,
            upper,
            MODEL_NAME,
            round(estimated_mape, 1),
            round(estimated_mape * avg_demand / 100, 1),  # MAE approx
            1.5,  # MASE > 1 = worse than naive (being honest)
            False,  # NOT reliable — these are extrapolations
            data['top_skills'],
            data['top_locations'],
            data['top_clients'],
        ))
        inserted += 1

conn.commit()

# ── Step 4: Verify ──
print(f"\n  Generated forecasts for {clusters_forecasted} clusters ({inserted} rows)")
print(f"  Skipped: {len(missing) - clusters_forecasted} (negligible demand)")

cur.execute("""
    SELECT data_type, model_used,
           COUNT(*) as row_count,
           COUNT(DISTINCT cluster_name) as clusters,
           SUM(demand_predicted) as total_demand
    FROM demand_forecasts
    GROUP BY data_type, model_used
    ORDER BY data_type, model_used
""")

print(f"\n{'='*70}")
print("UPDATED DEMAND_FORECASTS SUMMARY")
print("=" * 70)
print(f"  {'Type':<10} {'Model':<25} {'Rows':<8} {'Clusters':<10} {'Demand'}")
print(f"  {'-'*65}")
for r in cur.fetchall():
    print(f"  {r['data_type']:<10} {r['model_used'] or 'N/A':<25} {r['row_count']:<8} {r['clusters']:<10} {int(r['total_demand'])}")

# Per-quarter check
cur.execute("""
    SELECT 
        CASE 
            WHEN forecast_month BETWEEN '2026-01' AND '2026-03' THEN 'Q1'
            WHEN forecast_month BETWEEN '2026-04' AND '2026-06' THEN 'Q2'
            WHEN forecast_month BETWEEN '2026-07' AND '2026-09' THEN 'Q3'
            WHEN forecast_month BETWEEN '2026-10' AND '2026-12' THEN 'Q4'
        END as quarter,
        COUNT(DISTINCT cluster_name) as clusters,
        SUM(demand_predicted) as demand
    FROM demand_forecasts
    GROUP BY quarter
    ORDER BY quarter
""")
print(f"\n  Per-quarter cluster coverage:")
print(f"  {'Quarter':<10} {'Clusters':<10} {'Demand'}")
print(f"  {'-'*35}")
for r in cur.fetchall():
    print(f"  {r['quarter']:<10} {r['clusters']:<10} {int(r['demand'])}")

cur.close()
conn.close()

print(f"\n✅ P0 #2 Done — Missing clusters now have predictions (marked is_reliable=False)")
print(f"   Dashboard will show consistent clusters across all quarters")
