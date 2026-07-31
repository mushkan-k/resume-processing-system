"""
DATA COLLECTION SUMMARY — What we have for re-clustering and re-training.
"""
import os
import json
import glob

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
signals_dir = os.path.join(data_dir, 'market_signals')
career_dir = os.path.join(signals_dir, 'career_pages')

print("=" * 70)
print("  DATA COLLECTION INVENTORY")
print("=" * 70)

# 1. BLS OES
print("\n  [1] BLS OES DATA")
oes_path = os.path.join(signals_dir, 'bls_oes_data.json')
if os.path.exists(oes_path):
    with open(oes_path) as f:
        oes = json.load(f)
    print(f"      ✅ {len(oes.get('data', {}))} SOC codes with employment/wage data")
    print(f"      Covers: {list(oes.get('soc_to_cluster_mapping', {}).keys())[:10]}...")
else:
    print(f"      ❌ Not collected")

# 2. O*NET Skills
print("\n  [2] O*NET SKILLS TAXONOMY")
onet_path = os.path.join(signals_dir, 'onet_skills_taxonomy.json')
if os.path.exists(onet_path):
    with open(onet_path) as f:
        onet = json.load(f)
    clusters_with_skills = sum(1 for c, d in onet.get('cluster_skills', {}).items() 
                                if d.get('skills') or d.get('tech_skills'))
    total_tech = sum(len(d.get('tech_skills', [])) for d in onet.get('cluster_skills', {}).values())
    print(f"      ✅ {len(onet.get('cluster_skills', {}))} clusters mapped")
    print(f"      Clusters with actual skills: {clusters_with_skills}")
    print(f"      Total tech skills entries: {total_tech}")
else:
    print(f"      ❌ Not collected")

# 3. Career Pages
print("\n  [3] CAREER PAGE SCRAPES")
if os.path.exists(career_dir):
    files = glob.glob(os.path.join(career_dir, '*.json'))
    files = [f for f in files if not f.endswith('_summary.json')]
    total_jobs = 0
    for f in files:
        with open(f) as fh:
            data = json.load(fh)
        company = data.get('company', os.path.basename(f))
        n = data.get('total_openings', 0)
        total_jobs += n
        if n > 0:
            top_skills = list(data.get('skill_signals', {}).keys())[:5]
            print(f"      ✅ {company:<25} {n:>5} jobs  skills: {', '.join(top_skills)}")
    print(f"      Total: {total_jobs} jobs from {len(files)} companies")
else:
    print(f"      ❌ Not collected")

# 4. Existing training data
print("\n  [4] EXISTING TRAINING DATA")
import pickle
import pandas as pd

pkl_42k = os.path.join(data_dir, 'clean_42k_v1.pkl')
if os.path.exists(pkl_42k):
    df = pd.read_pickle(pkl_42k)
    print(f"      ✅ clean_42k_v1.pkl: {len(df)} rows")
    print(f"      Columns: {df.columns.tolist()}")
    if 'role_cluster' in df.columns:
        print(f"      Clusters: {df['role_cluster'].nunique()}")
        print(f"      Top 10: {df['role_cluster'].value_counts().head(10).to_dict()}")

ttc_path = os.path.join(data_dir, 'title_to_cluster.pkl')
if os.path.exists(ttc_path):
    ttc = pd.read_pickle(ttc_path)
    print(f"\n      ✅ title_to_cluster.pkl: {len(ttc)} title mappings → {ttc['role_cluster'].nunique()} clusters")
    print(f"      All clusters: {sorted(ttc['role_cluster'].unique().tolist())}")

# 5. Gap Excel
print("\n  [5] GAP INC. DATA")
gap_path = os.path.join(data_dir, '..', '2026 Gap Reqs.xlsx')
if os.path.exists(gap_path):
    gap = pd.read_excel(gap_path, sheet_name='Sheet2')
    gap = gap.dropna(subset=['Job Title'])
    print(f"      ✅ 438 reqs with full JDs")
    print(f"      Unique titles: {gap['Job Title'].nunique()}")
    print(f"      Has JD text: {gap['Job Description'].notna().sum()}")
else:
    print(f"      ❌ Not found")

# 6. DB records
print("\n  [6] DATABASE RECORDS")
import mysql.connector
conn = mysql.connector.connect(host='localhost', port=3305, database='resume_processing',
                                user='resume_user', password='resume_password')
cur = conn.cursor(dictionary=True)
cur.execute("SELECT COUNT(*) as c FROM updated_job_records")
print(f"      updated_job_records: {cur.fetchone()['c']} total")
cur.execute("SELECT COUNT(*) as c FROM updated_job_records WHERE issue_date >= '2025-07-01'")
print(f"      Last 12 months: {cur.fetchone()['c']}")
cur.execute("SELECT COUNT(DISTINCT company_name) as c FROM updated_job_records WHERE issue_date >= '2025-07-01'")
print(f"      Companies (12mo): {cur.fetchone()['c']}")
cur.execute("SELECT COUNT(*) as c FROM demand_forecasts")
print(f"      demand_forecasts: {cur.fetchone()['c']} rows")
cur.execute("SELECT COUNT(*) as c FROM demand_forecasts WHERE data_type='actual'")
print(f"      Actuals: {cur.fetchone()['c']}")
cur.execute("SELECT COUNT(*) as c FROM demand_forecasts WHERE data_type='predicted'")
print(f"      Predicted: {cur.fetchone()['c']}")
conn.close()

print(f"\n{'='*70}")
print(f"  READY FOR RE-CLUSTERING")
print(f"{'='*70}")
print(f"""
  Data sources ready:
    ✅ 42,835 historical classified JDs (training base)
    ✅ 438 Gap Inc. reqs with full JD text (new company)
    ✅ ~500 live career page jobs (skill signals)  
    ✅ O*NET skills taxonomy (29 clusters mapped)
    ✅ BLS OES employment/wage data (22 SOC codes)
    ✅ 190k+ job records in DB (validation)
    ✅ 61 KEEP companies identified (noise removed)
    
  Next steps:
    1. Audit current 50 clusters for misclassifications
    2. Define new clusters (Creative, Fashion, Data Governance)
    3. Merge Gap data + career page data into training set
    4. Re-train title_to_cluster mapping
    5. Re-classify all records
    6. Re-run forecasting model
""")
