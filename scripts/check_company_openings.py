"""
Break down actual openings by COMPANY for Q1 and Q2 2026.
Focus on T-Mobile to validate against demand_forecasts.
"""
import mysql.connector
import pandas as pd

DB_CONFIG = {
    "host": "localhost",
    "port": 3305,
    "database": "resume_processing",
    "user": "resume_user",
    "password": "resume_password",
}

conn = mysql.connector.connect(**DB_CONFIG)

# ─── Q1 + Q2 actuals by company ───
df = pd.read_sql("""
    SELECT company_name, country, title, role_cluster,
           DATE_FORMAT(issue_date, '%%Y-%%m') as month,
           openings, fills, issue_date
    FROM updated_job_records
    WHERE issue_date >= '2026-01-01' AND issue_date < '2026-07-01'
""", conn)

print("=" * 70)
print("  ACTUAL OPENINGS BY COMPANY — Q1 + Q2 2026")
print("=" * 70)

# Company totals
company_totals = df.groupby('company_name').agg(
    total_openings=('openings', 'sum'),
    job_count=('title', 'count'),
).sort_values('total_openings', ascending=False)

print(f"\n  TOP 20 COMPANIES (Jan-Jun 2026):")
print(f"  {'Company':<40} {'Openings':>10} {'Jobs':>8}")
print(f"  {'-'*40} {'-'*10} {'-'*8}")
for comp, row in company_totals.head(20).iterrows():
    print(f"  {comp:<40} {row['total_openings']:>10} {row['job_count']:>8}")
print(f"\n  Total companies: {len(company_totals)}")
print(f"  Total openings: {company_totals['total_openings'].sum()}")

# ─── T-Mobile deep dive ───
print(f"\n{'='*70}")
print("  T-MOBILE DEEP DIVE")
print(f"{'='*70}")

# Find T-Mobile (case-insensitive)
tmobile_mask = df['company_name'].str.lower().str.contains('t-mobile|tmobile|t mobile', na=False)
tmobile = df[tmobile_mask]

if tmobile.empty:
    # Try broader search
    print("  No exact 'T-Mobile' match. Searching similar...")
    for comp in company_totals.index:
        if 'mobile' in comp.lower() or 'tmob' in comp.lower():
            print(f"    Found: '{comp}'")
else:
    print(f"\n  Company name(s): {tmobile['company_name'].unique().tolist()}")
    print(f"  Total records: {len(tmobile)}")
    print(f"  Total openings: {tmobile['openings'].sum()}")

    # Monthly breakdown
    monthly = tmobile.groupby('month')['openings'].sum()
    print(f"\n  MONTHLY OPENINGS:")
    for month, opens in monthly.items():
        print(f"    {month}: {opens}")

    # Q1 vs Q2
    q1 = tmobile[tmobile['issue_date'] < '2026-04-01']['openings'].sum()
    q2 = tmobile[tmobile['issue_date'] >= '2026-04-01']['openings'].sum()
    print(f"\n  Q1 total: {q1}")
    print(f"  Q2 total: {q2}")

    # By cluster
    by_cluster = tmobile.groupby('role_cluster').agg(
        openings=('openings', 'sum'),
        jobs=('title', 'count'),
    ).sort_values('openings', ascending=False)
    print(f"\n  BY ROLE CLUSTER:")
    print(f"  {'Cluster':<40} {'Openings':>10} {'Jobs':>8}")
    print(f"  {'-'*40} {'-'*10} {'-'*8}")
    for cluster, row in by_cluster.iterrows():
        print(f"  {cluster:<40} {row['openings']:>10} {row['jobs']:>8}")

    # Individual job titles
    print(f"\n  ALL T-MOBILE JOB TITLES (Q2):")
    q2_jobs = tmobile[tmobile['issue_date'] >= '2026-04-01'].sort_values('issue_date')
    for _, row in q2_jobs.iterrows():
        print(f"    {row['issue_date'].strftime('%Y-%m-%d')} | {row['title']:<50} | opens={row['openings']} | {row['role_cluster'] or 'UNMAPPED'}")

# ─── Compare with demand_forecasts ───
print(f"\n{'='*70}")
print("  DEMAND_FORECASTS TABLE — T-MOBILE ENTRIES")
print(f"{'='*70}")

forecasts = pd.read_sql("""
    SELECT cluster_name, forecast_month, demand_predicted, data_type, 
           is_reliable, mape, top_clients
    FROM demand_forecasts
    WHERE top_clients LIKE '%%T-Mobile%%' OR top_clients LIKE '%%TMobile%%' OR top_clients LIKE '%%t-mobile%%'
    ORDER BY forecast_month
""", conn)

if forecasts.empty:
    # Check company_cluster_profiles instead
    print("  No direct T-Mobile in demand_forecasts.top_clients")
    print("\n  Checking company_cluster_profiles...")
    profiles = pd.read_sql("""
        SELECT company, cluster_name, demand_share, total_demand
        FROM company_cluster_profiles
        WHERE company LIKE '%%T-Mobile%%' OR company LIKE '%%TMobile%%' OR company LIKE '%%t-mobile%%'
        ORDER BY total_demand DESC
    """, conn)
    if not profiles.empty:
        print(f"  Found {len(profiles)} profile entries:")
        for _, row in profiles.iterrows():
            print(f"    {row['company']:<20} | {row['cluster_name']:<35} | share={row['demand_share']:.1%} | demand={row['total_demand']}")
    else:
        print("  No T-Mobile entries found in company_cluster_profiles either")
        # List all companies in profiles
        all_comps = pd.read_sql("SELECT DISTINCT company FROM company_cluster_profiles ORDER BY company", conn)
        print(f"\n  Available companies in profiles: {all_comps['company'].tolist()[:30]}")
else:
    print(f"  Found {len(forecasts)} forecast rows mentioning T-Mobile:")
    for _, row in forecasts.iterrows():
        print(f"    {row['forecast_month']} | {row['cluster_name']:<35} | demand={row['demand_predicted']} | type={row['data_type']} | clients={row['top_clients'][:60]}")

conn.close()
