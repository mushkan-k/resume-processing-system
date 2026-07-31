"""Sanity check: is the forecast inflated vs historical actuals?"""
import pandas as pd
import mysql.connector
import numpy as np

conn = mysql.connector.connect(
    host='localhost', port=3305,
    user='resume_user', password='resume_password',
    database='resume_processing'
)
cur = conn.cursor(dictionary=True)

# 1. Forecast totals per month
cur.execute("""SELECT forecast_month, SUM(demand_predicted) as total_pred, COUNT(*) as n_rows
              FROM demand_forecasts WHERE data_type = 'predicted'
              GROUP BY forecast_month ORDER BY forecast_month""")
print("=== FORECAST IN DB (per month) ===")
for r in cur.fetchall():
    print(f"  {r['forecast_month']}: {r['total_pred']:,.0f} openings ({r['n_rows']} cluster-rows)")

# 2. Historical actuals
cur.execute("""SELECT forecast_month, SUM(demand_predicted) as total
              FROM demand_forecasts WHERE data_type = 'actual'
              GROUP BY forecast_month ORDER BY forecast_month DESC LIMIT 12""")
rows = cur.fetchall()
print("\n=== RECENT ACTUALS (per month) ===")
for r in rows:
    print(f"  {r['forecast_month']}: {r['total']:,.0f} openings")

# 3. Check Developer specifically
cur.execute("""SELECT forecast_month, demand_predicted, demand_lower, demand_upper
              FROM demand_forecasts 
              WHERE cluster_name = 'Developer' AND data_type = 'predicted'
              ORDER BY forecast_month""")
print("\n=== DEVELOPER FORECAST ===")
for r in cur.fetchall():
    print(f"  {r['forecast_month']}: {r['demand_predicted']} (range {r['demand_lower']}-{r['demand_upper']})")

conn.close()

# 4. Also check directly from model output - what does the aggregation look like?
print("\n=== CHECKING AGGREGATION LOGIC ===")
df = pd.read_pickle('data/clean_42k_v1.pkl')
ttc = pd.read_pickle('data/title_to_cluster.pkl')
title_map = dict(zip(ttc['raw_title'], ttc['role_cluster']))
df['role_cluster'] = df['title'].map(title_map)
df = df.dropna(subset=['role_cluster'])
df['month'] = pd.to_datetime(df['issue_date']).dt.to_period('M')

# What are the ACTUAL monthly totals across all clusters from raw data?
monthly = df.groupby('month')['openings'].sum().sort_index()
print("\nActual monthly totals (last 12 months of raw data):")
for m, v in monthly.tail(12).items():
    print(f"  {m}: {v:,.0f}")

print(f"\n  Avg monthly total (last 6 real months): {monthly.tail(7).head(6).mean():,.0f}")
print(f"  Expected 6-month forecast total: ~{monthly.tail(7).head(6).sum():,.0f}")

# 5. How many combos feed into Developer?
dev = df[df['role_cluster'] == 'Developer']
dev_monthly = dev.groupby('month')['openings'].sum().sort_index()
print(f"\n=== DEVELOPER ACTUAL (last 6 months) ===")
for m, v in dev_monthly.tail(7).items():
    print(f"  {m}: {v:,.0f}")
