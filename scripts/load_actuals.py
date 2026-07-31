"""
Build unified demand data: Actuals (Q1/Q2) + Predictions (Q3/Q4)
================================================================
The dashboard needs all 4 quarters of 2026 in the same format:
- Q1 (Jan-Mar): ACTUAL data from updated_job_records  
- Q2 (Apr-Jun): ACTUAL data from updated_job_records
- Q3 (Jul-Sep): PREDICTED from demand_forecasts model
- Q4 (Oct-Dec): PREDICTED from demand_forecasts model

This script:
1. Loads actuals from updated_job_records + maps to role clusters
2. Inserts them into demand_forecasts with a flag (data_type = 'actual')
3. Updates the API to serve both in unified format
"""
import mysql.connector
import pandas as pd
import numpy as np
import json
import os

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

DB_CONFIG = {
    "host": "localhost",
    "port": 3305,
    "database": "resume_processing",
    "user": "resume_user",
    "password": "resume_password",
}

# ─── City normalization (merge common spelling variants) ───
CITY_NORMALIZE = {
    "Bangalor": "Bangalore",
    "Bengaluru": "Bangalore",
    "BENGALURU": "Bangalore",
    "BANGALORE": "Bangalore",
    "bengaluru": "Bangalore",
    "Hydrabad": "Hyderabad",
    "HYDERABAD": "Hyderabad",
    "Chenai": "Chennai",
    "CHENNAI": "Chennai",
    "PUNE": "Pune",
    "MUMBAI": "Mumbai",
    "NOIDA": "Noida",
    "Gurugram": "Gurgaon",
    "GURUGRAM": "Gurgaon",
}


def normalize_location_list(locations):
    """Normalize and deduplicate location names."""
    if not locations:
        return []
    seen = set()
    result = []
    for loc in locations:
        loc = loc.strip()
        if not loc:
            continue
        normalized = CITY_NORMALIZE.get(loc, loc)
        if normalized.lower() not in seen:
            seen.add(normalized.lower())
            result.append(normalized)
    return result

# ═══════════════════════════════════════════════════════════════
# STEP 1: Load role cluster mapping
# ═══════════════════════════════════════════════════════════════
mapping = pd.read_pickle(os.path.join(data_dir, 'title_to_cluster.pkl'))
title_to_role = dict(zip(mapping['raw_title'], mapping['role_cluster']))

print("=" * 70)
print("BUILDING UNIFIED ACTUALS + PREDICTIONS")
print("=" * 70)
print(f"\n  Role cluster mapping: {len(title_to_role)} titles → {len(set(title_to_role.values()))} clusters")

# ═══════════════════════════════════════════════════════════════
# STEP 2: Add data_type column to demand_forecasts if not exists
# ═══════════════════════════════════════════════════════════════
conn = mysql.connector.connect(**DB_CONFIG)
cur = conn.cursor(dictionary=True)

# Add columns if needed
try:
    cur.execute("ALTER TABLE demand_forecasts ADD COLUMN data_type VARCHAR(20) DEFAULT 'predicted'")
    conn.commit()
    print("  Added data_type column")
except Exception as e:
    if "Duplicate column" in str(e):
        print("  data_type column already exists")
    else:
        print(f"  Column add note: {e}")

# Mark all existing forecasts as 'predicted'
cur.execute("UPDATE demand_forecasts SET data_type = 'predicted' WHERE data_type IS NULL")
conn.commit()

# ═══════════════════════════════════════════════════════════════
# STEP 3: Get actual data from job_records for Q1+Q2 2026
# ═══════════════════════════════════════════════════════════════
cur.execute("""
    SELECT title, company_name, country, role_cluster as db_role_cluster,
           DATE_FORMAT(issue_date, '%Y-%m') as month,
           SUM(openings) as openings, 
           SUM(fills) as fills,
           COUNT(*) as job_count,
           GROUP_CONCAT(DISTINCT state ORDER BY state SEPARATOR ', ') as locations
    FROM updated_job_records
    WHERE issue_date >= '2026-01-01' AND issue_date < '2026-07-01'
    GROUP BY title, company_name, country, role_cluster, DATE_FORMAT(issue_date, '%Y-%m')
""")
actual_rows = cur.fetchall()
print(f"\n  Actual updated_job_records (Jan-Jun 2026): {len(actual_rows)} title/company/month combos")

# Map titles to role clusters
actuals_mapped = []
unmapped_count = 0
used_db_cluster = 0
used_pkl_cluster = 0
for row in actual_rows:
    title = row['title']
    
    # Priority 1: Use pre-assigned role_cluster from DB
    role_cluster = row.get('db_role_cluster')
    if role_cluster:
        used_db_cluster += 1
    else:
        # Priority 2: Use pickle mapping
        role_cluster = title_to_role.get(title)
        if not role_cluster:
            # Try partial matching
            for raw_t, cluster in title_to_role.items():
                if raw_t.lower() in title.lower() or title.lower() in raw_t.lower():
                    role_cluster = cluster
                    break
        if role_cluster:
            used_pkl_cluster += 1
    
    if not role_cluster:
        unmapped_count += 1
        continue
    
    # Build combo_key (same as forecasts)
    region = row['country'] if row['country'] else 'US'
    combo_key = f"{region} | {role_cluster}"
    
    actuals_mapped.append({
        'cluster_name': combo_key,
        'month': row['month'],
        'openings': int(row['openings']),
        'fills': int(row['fills']),
        'company': row['company_name'],
        'locations': row['locations'],
    })

print(f"  Mapped: {len(actuals_mapped)} rows (DB cluster: {used_db_cluster}, Pickle: {used_pkl_cluster})")
print(f"  Unmapped: {unmapped_count}")

# Aggregate by cluster + month
actuals_df = pd.DataFrame(actuals_mapped)
actuals_agg = actuals_df.groupby(['cluster_name', 'month']).agg(
    demand=('openings', 'sum'),
    fills=('fills', 'sum'),
    companies=('company', lambda x: list(x.unique())[:5]),
    locations=('locations', lambda x: normalize_location_list(list(set(','.join(x.dropna()).split(', ')))[:8])),
).reset_index()

print(f"  Aggregated: {len(actuals_agg)} cluster/month combos")
print(f"  Clusters represented: {actuals_agg['cluster_name'].nunique()}")
print(f"  Months: {sorted(actuals_agg['month'].unique())}")

# ═══════════════════════════════════════════════════════════════
# STEP 4: Insert actuals into demand_forecasts
# ═══════════════════════════════════════════════════════════════

# Clear existing actuals (in case we re-run)
cur.execute("DELETE FROM demand_forecasts WHERE data_type = 'actual'")
deleted = cur.rowcount
print(f"\n  Cleared {deleted} existing actual rows")

# Remove predicted rows for months that are now actual (Jan-Jun 2026)
cur.execute("DELETE FROM demand_forecasts WHERE data_type = 'predicted' AND forecast_month BETWEEN '2026-01' AND '2026-06'")
deleted_pred = cur.rowcount
if deleted_pred:
    print(f"  Removed {deleted_pred} predicted rows for months now covered by actuals")
conn.commit()

# Get existing forecast metadata (skills, mape, etc.) per cluster
cur.execute("""
    SELECT DISTINCT cluster_name, mape, mae, mase, is_reliable, 
           top_skills, top_locations, top_clients
    FROM demand_forecasts WHERE data_type = 'predicted'
""")
forecast_meta = {r['cluster_name']: r for r in cur.fetchall()}

# Insert actuals
insert_sql = """
    INSERT INTO demand_forecasts 
    (cluster_name, forecast_month, demand_predicted, demand_lower, demand_upper,
     model_used, mape, mae, mase, is_reliable, top_skills, top_locations, top_clients, data_type)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'actual')
"""

inserted = 0
for _, row in actuals_agg.iterrows():
    cluster = row['cluster_name']
    month = row['month']
    demand = int(row['demand'])
    
    # Get metadata from forecast for same cluster (if exists)
    meta = forecast_meta.get(cluster, {})
    
    # For actuals, lower=upper=demand (no uncertainty — it's real data)
    cur.execute(insert_sql, (
        cluster,
        month,
        demand,
        demand,  # lower = actual (no interval for actuals)
        demand,  # upper = actual
        'Actual',
        meta.get('mape'),
        meta.get('mae'),
        meta.get('mase'),
        meta.get('is_reliable', True),
        meta.get('top_skills'),
        meta.get('top_locations'),
        meta.get('top_clients'),
    ))
    inserted += 1

conn.commit()
print(f"  Inserted {inserted} actual rows")

# ═══════════════════════════════════════════════════════════════
# STEP 5: Verify the unified data
# ═══════════════════════════════════════════════════════════════
cur.execute("""
    SELECT data_type, 
           MIN(forecast_month) as from_month, 
           MAX(forecast_month) as to_month,
           COUNT(*) as row_count,
           SUM(demand_predicted) as total_demand,
           COUNT(DISTINCT cluster_name) as clusters
    FROM demand_forecasts
    GROUP BY data_type
""")
summary = cur.fetchall()

print(f"\n{'='*70}")
print("UNIFIED DATA SUMMARY")
print("=" * 70)
for s in summary:
    print(f"  {s['data_type']:<10}: {s['from_month']} → {s['to_month']}, "
          f"{s['row_count']} rows, {s['clusters']} clusters, {int(s['total_demand'])} demand")

# Show per-quarter breakdown
cur.execute("""
    SELECT 
        CASE 
            WHEN forecast_month BETWEEN '2026-01' AND '2026-03' THEN '2026-Q1'
            WHEN forecast_month BETWEEN '2026-04' AND '2026-06' THEN '2026-Q2'
            WHEN forecast_month BETWEEN '2026-07' AND '2026-09' THEN '2026-Q3'
            WHEN forecast_month BETWEEN '2026-10' AND '2026-12' THEN '2026-Q4'
        END as quarter,
        data_type,
        SUM(demand_predicted) as demand,
        COUNT(DISTINCT cluster_name) as clusters
    FROM demand_forecasts
    GROUP BY quarter, data_type
    ORDER BY quarter
""")
quarters = cur.fetchall()

print(f"\n  Per-quarter breakdown:")
print(f"  {'Quarter':<10} {'Type':<12} {'Demand':>8} {'Clusters':>9}")
print(f"  {'─'*42}")
for q in quarters:
    print(f"  {q['quarter']:<10} {q['data_type']:<12} {int(q['demand']):>8} {q['clusters']:>9}")

cur.close()
conn.close()

print(f"\n✅ Done! Database now has Q1/Q2 actuals + Q3/Q4 predictions")
print(f"   The API needs to surface data_type field so frontend can distinguish")
