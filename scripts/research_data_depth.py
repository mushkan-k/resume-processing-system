"""Research: Where can we get more data to improve model accuracy?"""
import pandas as pd
import numpy as np

df = pd.read_pickle('data/clean_42k_v1.pkl')
ttc = pd.read_pickle('data/title_to_cluster.pkl')
title_map = dict(zip(ttc['raw_title'], ttc['role_cluster']))
df['role_cluster'] = df['title'].map(title_map)
df = df.dropna(subset=['role_cluster'])
df['month'] = pd.to_datetime(df['issue_date']).dt.to_period('M')

print("=" * 70)
print("  DATA DEPTH RESEARCH")
print("=" * 70)

# 1. Date range
print("\n--- 1. CURRENT DATA RANGE ---")
print(f"  Earliest record: {df['issue_date'].min()}")
print(f"  Latest record:   {df['issue_date'].max()}")
print(f"  Total records: {len(df):,}")
print(f"  Total months span: {df['month'].nunique()}")

# 2. Monthly volume ramp
monthly = df.groupby('month')['openings'].sum().sort_index()
print("\n--- 2. MONTHLY VOLUME (is there older data we're missing?) ---")
print("  First 12 months:")
for m, v in monthly.head(12).items():
    print(f"    {m}: {v:,.0f} openings")
print("  ...")
print("  Last 6 months:")
for m, v in monthly.tail(6).items():
    print(f"    {m}: {v:,.0f} openings")

# 3. Role-level depth (ignoring company dimension)
print("\n--- 3. ROLE-LEVEL DEPTH (aggregate across companies) ---")
role_months = df.groupby(['role_cluster', 'month']).agg(openings=('openings', 'sum')).reset_index()
role_depth = role_months.groupby('role_cluster')['month'].nunique().sort_values(ascending=False)
print(f"  Avg months per role: {role_depth.mean():.1f}")
print(f"  Roles with 80+ months: {(role_depth >= 80).sum()} / {len(role_depth)}")
print(f"  Roles with 60+ months: {(role_depth >= 60).sum()} / {len(role_depth)}")
print(f"  Top 10:")
for role, n in role_depth.head(10).items():
    print(f"    {n:3d} mo | {role}")

# 4. Zero-fill potential
print("\n--- 4. ZERO-FILL GAP ANALYSIS ---")
ts = df.groupby(['role_cluster', 'company_name', 'month']).agg(openings=('openings', 'sum')).reset_index()
combo_stats = ts.groupby(['role_cluster', 'company_name']).agg(
    first_month=('month', 'min'),
    last_month=('month', 'max'),
    active_months=('month', 'nunique')
).reset_index()
combo_stats['span'] = (combo_stats['last_month'] - combo_stats['first_month']).apply(lambda x: x.n + 1)
combo_stats['gap_months'] = combo_stats['span'] - combo_stats['active_months']
viable = combo_stats[combo_stats['active_months'] >= 20]
print(f"  Viable combos (>=20 active months): {len(viable)}")
print(f"  Avg active months: {viable['active_months'].mean():.1f}")
print(f"  Avg span (first to last): {viable['span'].mean():.1f}")
print(f"  Avg gap months that could be zero-filled: {viable['gap_months'].mean():.1f}")
print(f"  AFTER zero-fill, avg months per combo: {viable['span'].mean():.1f}")
print(f"  Improvement: +{viable['gap_months'].mean():.1f} months per combo ({viable['gap_months'].mean()/viable['active_months'].mean()*100:.0f}% more data)")

# 5. What about the DB? Check if there's older data there
print("\n--- 5. DATABASE CHECK (is there older data not in pickle?) ---")
try:
    import mysql.connector
    conn = mysql.connector.connect(
        host='localhost', port=3305,
        user='resume_user', password='resume_password',
        database='resume_processing'
    )
    cur = conn.cursor()
    cur.execute("SELECT MIN(issue_date), MAX(issue_date), COUNT(*) FROM jobs")
    mn, mx, cnt = cur.fetchone()
    print(f"  DB jobs table: {mn} to {mx}, {cnt:,} records")
    cur.execute("SELECT MIN(issue_date), MAX(issue_date), COUNT(*) FROM jobs WHERE issue_date < '2017-03-01'")
    mn2, mx2, cnt2 = cur.fetchone()
    print(f"  Records BEFORE pickle start (2017-03): {cnt2 or 0:,}")
    conn.close()
except Exception as e:
    print(f"  DB not accessible: {e}")

# 6. JobDiva API — how far back can we query?
print("\n--- 6. JOBDIVA API HISTORICAL DEPTH ---")
print("  Current pickle starts: 2017-03")
print("  JobDiva typically retains: 7-10 years of job history")
print("  Potential: Query 2014-01 to 2017-02 = 38 additional months")
print("  If available, best combos could go from 64 -> 100+ months")

# 7. Alternative: model at ROLE level (not role x company)
print("\n--- 7. ALTERNATIVE: ROLE-LEVEL MODEL (drop company dim) ---")
role_panel = role_months.copy()
role_panel = role_panel.sort_values(['role_cluster', 'month'])
role_depth_viable = role_depth[role_depth >= 20]
print(f"  Viable roles (>=20 mo): {len(role_depth_viable)} / {len(role_depth)}")
print(f"  Avg depth: {role_depth_viable.mean():.1f} months")
print(f"  Would give: fewer combos but MUCH deeper history per series")
print(f"  Panel rows: ~{role_depth_viable.sum()}")

# 8. Hybrid approach: use role-level as a feature
print("\n--- 8. RECOMMENDATION SUMMARY ---")
print("""
  OPTION A: Pull older JobDiva data (2014-2017)
    - Could add 38 months to existing combos
    - Best combos: 64 -> potentially 100+ months
    - Effort: Medium (API query, re-build pickle)
    - Impact: HIGH for long-lived combos

  OPTION B: Zero-fill gaps in existing data
    - Adds ~{:.0f} months per combo on average
    - Tells model 'no demand in these months' (real signal)
    - Effort: LOW (code change in panel builder)
    - Impact: MEDIUM (more data points, better seasonality)

  OPTION C: Add role-level aggregate as feature
    - Each combo gets role-total as context signal
    - Role-level has {:.0f} months avg depth
    - Effort: LOW (add column to panel)
    - Impact: MEDIUM (hierarchical already does this partially)

  OPTION D: Model at role-level only (drop company)
    - {:.0f} roles with avg {:.1f} months history
    - Much deeper but loses company granularity
    - Effort: LOW (change aggregation)
    - Impact: MAPE drops but forecast is less specific
""".format(
    viable['gap_months'].mean(),
    role_depth_viable.mean(),
    len(role_depth_viable),
    role_depth_viable.mean()
))
