"""Check what data sources are available vs what's in training."""
import pandas as pd
import mysql.connector
import os, json

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

# 1. Current pickle
df = pd.read_pickle(os.path.join(data_dir, 'clean_42k_v1.pkl'))
print('=== CURRENT TRAINING (clean_42k_v1.pkl) ===')
print(f'  Records: {len(df):,}')
print(f'  Date range: {df["issue_date"].min()} to {df["issue_date"].max()}')
print(f'  Companies: {df["company_name"].nunique()}')
print(f'  Columns: {df.columns.tolist()}')
print()

# 2. DB records
conn = mysql.connector.connect(host='localhost', port=3305, database='resume_processing',
                               user='resume_user', password='resume_password')
cur = conn.cursor(dictionary=True)
cur.execute('SELECT COUNT(*) as cnt, MIN(issue_date) as mn, MAX(issue_date) as mx, COUNT(DISTINCT company_name) as comps FROM updated_job_records')
r = cur.fetchone()
print('=== DB (updated_job_records) ===')
print(f'  Records: {r["cnt"]:,}, Date: {r["mn"]} to {r["mx"]}, Companies: {r["comps"]}')

# What's in DB but not in pickle?
pickle_max = str(df['issue_date'].max())[:10]
cur.execute('SELECT COUNT(*) as cnt FROM updated_job_records WHERE issue_date > %s', (pickle_max,))
r2 = cur.fetchone()
print(f'  Records AFTER pickle max date ({pickle_max}): {r2["cnt"]:,}')

# Gap specifically
cur.execute("SELECT COUNT(*) as cnt, MIN(issue_date) as mn, MAX(issue_date) as mx FROM updated_job_records WHERE company_name LIKE '%Gap%'")
r3 = cur.fetchone()
print(f'  Gap Inc in DB: {r3["cnt"]} records ({r3["mn"]} to {r3["mx"]})')

# Check if Gap is in pickle
gap_in_pkl = df[df['company_name'].str.contains('Gap', case=False, na=False)]
print(f'  Gap Inc in pickle: {len(gap_in_pkl)} records')

# 3. Career pages
cp_dir = os.path.join(data_dir, 'market_signals', 'career_pages')
if os.path.exists(cp_dir):
    total_jobs = 0
    companies = []
    for f in os.listdir(cp_dir):
        if f.endswith('.json'):
            data = json.load(open(os.path.join(cp_dir, f)))
            jobs = data.get('jobs', [])
            total_jobs += len(jobs)
            companies.append(f.replace('.json',''))
    print(f'\n=== CAREER PAGE SCRAPES ===')
    print(f'  Files: {len(companies)}, Total jobs: {total_jobs}')
    print(f'  Companies: {", ".join(companies[:10])}...')

# 4. Summary
print('\n=== WHAT SHOULD BE ADDED TO TRAINING ===')
print(f'  1. DB records after {pickle_max} (newer actuals): ~{r2["cnt"]:,} records')
print(f'  2. Gap Inc from DB: {r3["cnt"]} records (new company)')
print(f'  3. Career page scrapes: {total_jobs} jobs (for volume/trend signals)')
print(f'  TOTAL potential addition: significant boost to recent-period signal')

conn.close()
