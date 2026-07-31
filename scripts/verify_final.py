"""Quick verification of the full pipeline after re-clustering."""
import mysql.connector

conn = mysql.connector.connect(host='localhost', port=3305, database='resume_processing',
                                user='resume_user', password='resume_password')
cur = conn.cursor(dictionary=True)

print("=" * 70)
print("  FINAL VERIFICATION — POST RE-CLUSTERING")
print("=" * 70)

# Overall summary
cur.execute("""
    SELECT data_type, COUNT(*) as row_count, COUNT(DISTINCT cluster_name) as clusters,
           SUM(demand_predicted) as demand, MIN(forecast_month) as from_m, MAX(forecast_month) as to_m
    FROM demand_forecasts GROUP BY data_type
""")
for r in cur.fetchall():
    print(f"\n  {r['data_type']}: {r['row_count']} rows, {r['clusters']} clusters, {int(r['demand'])} demand ({r['from_m']} → {r['to_m']})")

# Quarter breakdown
cur.execute("""
    SELECT 
        CASE 
            WHEN forecast_month BETWEEN '2026-01' AND '2026-03' THEN 'Q1'
            WHEN forecast_month BETWEEN '2026-04' AND '2026-06' THEN 'Q2'
            WHEN forecast_month BETWEEN '2026-07' AND '2026-09' THEN 'Q3'
            WHEN forecast_month BETWEEN '2026-10' AND '2026-12' THEN 'Q4'
        END as quarter,
        data_type, SUM(demand_predicted) as demand, COUNT(DISTINCT cluster_name) as clusters
    FROM demand_forecasts GROUP BY quarter, data_type ORDER BY quarter
""")
print(f"\n  {'Quarter':<8} {'Type':<12} {'Demand':>8} {'Clusters':>9}")
print(f"  {'-'*40}")
for r in cur.fetchall():
    print(f"  {r['quarter']:<8} {r['data_type']:<12} {int(r['demand']):>8} {r['clusters']:>9}")

# T-Mobile specific check
cur.execute("""
    SELECT cluster_name, forecast_month, demand_predicted, data_type
    FROM demand_forecasts
    WHERE top_clients LIKE '%%T-Mobile%%'
    ORDER BY forecast_month
    LIMIT 10
""")
tmobile = cur.fetchall()
print(f"\n  T-Mobile in forecasts: {len(tmobile)} rows")
for r in tmobile[:5]:
    print(f"    {r['forecast_month']} | {r['cluster_name']:<35} | {int(r['demand_predicted'])} ({r['data_type']})")

# New clusters check
cur.execute("""
    SELECT cluster_name, SUM(demand_predicted) as demand, data_type
    FROM demand_forecasts
    WHERE cluster_name IN ('Photo & Styling','Fashion / Apparel Designer','Creative Director / Art Director','Content & Communications','Data Governance')
    GROUP BY cluster_name, data_type
""")
new_cl = cur.fetchall()
if new_cl:
    print(f"\n  NEW CLUSTERS in demand_forecasts:")
    for r in new_cl:
        print(f"    {r['cluster_name']:<40} demand={int(r['demand'])} ({r['data_type']})")
else:
    print(f"\n  New clusters not yet in demand_forecasts (only in actuals from updated_job_records)")

# Check reliability distribution
cur.execute("""
    SELECT 
        CASE WHEN mape <= 35 THEN 'Very Stable'
             WHEN mape <= 50 THEN 'Stable'
             WHEN mape <= 70 THEN 'Moderate'
             ELSE 'Volatile' END as grade,
        COUNT(DISTINCT cluster_name) as clusters
    FROM demand_forecasts
    WHERE data_type = 'predicted' AND mape IS NOT NULL
    GROUP BY grade
""")
grades = cur.fetchall()
print(f"\n  GRADE DISTRIBUTION (predicted clusters):")
for r in grades:
    print(f"    {r['grade']:<15} {r['clusters']} clusters")

conn.close()
