"""
Fix fills column in training data using job_status from DB.
============================================================
Logic:
  FILLED  → fills = openings (Aditi placed someone)
  CLOSED  → fills = openings (position was filled by someone else / client)
  CANCELLED → fills = 0 (worked on it, didn't work out)
  OPEN    → fills = 0 (still active)
  Others  → fills = 0
"""
import pandas as pd
import mysql.connector
import os

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

print("=" * 70)
print("FIXING FILLS COLUMN USING JOB_STATUS")
print("=" * 70)

# Load training data
df = pd.read_pickle(os.path.join(data_dir, 'clean_42k_v1.pkl'))
print(f"\n  Training data: {len(df):,} records")
print(f"  Current zero fills: {(df.fills == 0).sum():,} ({(df.fills == 0).mean()*100:.1f}%)")

# Get status mapping from DB
conn = mysql.connector.connect(
    host='localhost', port=3305, database='resume_processing',
    user='resume_user', password='resume_password'
)
cur = conn.cursor(dictionary=True)

# Build a lookup: match on ALL available fields to avoid wrong matches
cur.execute("""
    SELECT title, company_name, issue_date, job_status, openings, fills,
           country, state, city, skills
    FROM updated_job_records
""")
rows = cur.fetchall()
print(f"  DB records loaded: {len(rows):,}")

# Build lookup by (title, company, date, openings, country, state)
# Using all fields ensures we match the EXACT same record
status_lookup = {}
for r in rows:
    # Full key: all columns that exist in both training + DB
    date_str = str(r['issue_date'])[:10]
    key_full = (r['title'], r['company_name'], date_str, int(r['openings']),
                str(r['country'] or ''), str(r['state'] or ''))
    status_lookup[key_full] = {
        'job_status': r['job_status'],
        'db_fills': r['fills'],
        'db_openings': r['openings'],
    }

print(f"  Status lookup entries: {len(status_lookup):,}")

# Apply status-based fills correction
fixed = 0
filled_fix = 0
closed_fix = 0
cancelled_fix = 0
no_match = 0

# Map training 'region' back to DB 'country' for matching
region_to_country = {'US': 'US', 'IN': 'IN', 'LATAM': '', 'OTHER': ''}

for idx, row in df.iterrows():
    date_str = str(row['issue_date'])[:10]
    openings = int(row['openings'])
    region = str(row.get('region', ''))
    location = str(row.get('location', ''))
    
    # Try full key first (most precise match)
    key_full = (row['title'], row['company_name'], date_str, openings, region, location)
    info = status_lookup.get(key_full)
    
    # Fallback: try with country=region, state=location
    if not info:
        for country_val in [region, '']:
            for state_val in [location, '']:
                key_try = (row['title'], row['company_name'], date_str, openings, country_val, state_val)
                info = status_lookup.get(key_try)
                if info:
                    break
            if info:
                break
    
    if not info:
        no_match += 1
        continue
    
    status = info['job_status']
    openings = int(row['openings'])
    
    if status == 'FILLED':
        # Aditi filled it — fills should equal openings
        if row['fills'] == 0:
            df.at[idx, 'fills'] = openings
            filled_fix += 1
            fixed += 1
    elif status == 'CLOSED':
        # Someone filled it (client or competitor) — position was filled
        if row['fills'] == 0:
            df.at[idx, 'fills'] = openings
            closed_fix += 1
            fixed += 1
    elif status in ('CANCELLED', 'IGNORED', 'EXPIRED'):
        # Not filled — keep fills = 0
        if row['fills'] > 0:
            df.at[idx, 'fills'] = 0
            cancelled_fix += 1
            fixed += 1
    # OPEN, PIPELINE, Pending Approval — leave as-is

print(f"\n  === FIXES APPLIED ===")
print(f"  FILLED with zero fills → set fills=openings: {filled_fix:,}")
print(f"  CLOSED with zero fills → set fills=openings: {closed_fix:,}")
print(f"  CANCELLED/IGNORED with fills>0 → set fills=0: {cancelled_fix:,}")
print(f"  Total fixes: {fixed:,}")
print(f"  No DB match (synthetic/gap-fill records): {no_match:,}")

print(f"\n  === AFTER FIX ===")
print(f"  Zero fills: {(df.fills == 0).sum():,} ({(df.fills == 0).mean()*100:.1f}%)")
print(f"  Has fills:  {(df.fills > 0).sum():,} ({(df.fills > 0).mean()*100:.1f}%)")
print(f"  Avg fills:  {df.fills.mean():.2f}")

# Save
df.to_pickle(os.path.join(data_dir, 'clean_42k_v1.pkl'))
print(f"\n  [OK] Saved updated training data")

conn.close()
