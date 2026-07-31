"""
Cross-reference Client list with actual openings. Filter out companies <= 5 openings.
"""
import pandas as pd
import mysql.connector
import os

file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Client (1).xlsx')
df_clients = pd.read_excel(file_path, sheet_name='Export')

# Get all companies from the client list
client_list = df_clients['Client Name'].dropna().tolist()
print(f"{'='*70}")
print(f"  CLIENT LIST vs ACTUAL OPENINGS (Jan-Jun 2026)")
print(f"{'='*70}")
print(f"\n  Total clients in list: {len(client_list)}")

# Get actuals from DB
DB_CONFIG = {
    "host": "localhost", "port": 3305, "database": "resume_processing",
    "user": "resume_user", "password": "resume_password",
}
conn = mysql.connector.connect(**DB_CONFIG)

df_jobs = pd.read_sql("""
    SELECT company_name, COUNT(*) as job_count, SUM(openings) as total_openings,
           COUNT(DISTINCT title) as unique_titles,
           MIN(issue_date) as first_date, MAX(issue_date) as last_date
    FROM updated_job_records
    WHERE issue_date >= '2025-07-01'
    GROUP BY company_name
    ORDER BY total_openings DESC
""", conn)
conn.close()

# Match client list to DB
print(f"\n  Companies in DB (last 12 months): {len(df_jobs)}")
print(f"\n{'='*70}")
print(f"  ALL CLIENTS — RANKED BY OPENINGS (last 12 months)")
print(f"{'='*70}")
print(f"  {'#':<4} {'Client':<35} {'Openings':>10} {'Jobs':>8} {'Titles':>8} {'Status'}")
print(f"  {'-'*4} {'-'*35} {'-'*10} {'-'*8} {'-'*8} {'-'*12}")

keep = []
drop = []

for i, (_, row) in enumerate(df_clients.iterrows()):
    client = row['Client Name']
    positions = row.get('Positions', 0)
    # Find in DB
    match = df_jobs[df_jobs['company_name'].str.lower() == str(client).lower()]
    if not match.empty:
        opens = int(match.iloc[0]['total_openings'])
        jobs = int(match.iloc[0]['job_count'])
        titles = int(match.iloc[0]['unique_titles'])
    else:
        opens = int(positions) if pd.notna(positions) else 0
        jobs = 0
        titles = 0
    
    status = "✅ KEEP" if opens > 5 else "❌ DROP"
    if opens > 5:
        keep.append(client)
    else:
        drop.append(client)
    
    print(f"  {i+1:<4} {str(client):<35} {opens:>10} {jobs:>8} {titles:>8} {status}")

print(f"\n{'='*70}")
print(f"  SUMMARY")
print(f"{'='*70}")
print(f"  KEEP (>5 openings): {len(keep)} companies")
print(f"  DROP (<=5 openings): {len(drop)} companies")
print(f"\n  Companies to KEEP:")
for c in keep:
    print(f"    ✅ {c}")
print(f"\n  Companies to DROP (noise):")
for c in drop:
    print(f"    ❌ {c}")

# Check if Gap Inc is in the list
print(f"\n{'='*70}")
print(f"  GAP INC CHECK")
print(f"{'='*70}")
gap_match = [c for c in client_list if 'gap' in str(c).lower()]
print(f"  Gap in client list: {gap_match if gap_match else 'NOT FOUND — new company to add'}")
