"""
Build Company-Specific Skill Profiles per Cluster
===================================================
Problem: top_skills in demand_forecasts is cluster-level (all companies mixed).
         When filtering by Wabtec, you see Python/Java/AWS from PayPal/Intel.
         Wabtec actually needs Embedded C, C++, RTOS.

Solution: Pre-compute per-company per-cluster skill profiles from raw JD data.
          Store in a new table. API uses company-specific skills when ?company= is set.
"""
import pandas as pd
import numpy as np
import re
import json
import mysql.connector

# ─── Load data ───
df = pd.read_pickle('data/clean_42k_v1.pkl')
df['issue_date'] = pd.to_datetime(df['issue_date'])

mapping = pd.read_pickle('data/title_to_cluster.pkl')
title_to_role = dict(zip(mapping['raw_title'], mapping['role_cluster']))
df['role_cluster'] = df['title'].map(title_to_role)

# Build cluster_name = region | role_cluster
df['cluster_name'] = df['region'] + ' | ' + df['role_cluster']


def parse_skills(s):
    """Extract skills from boolean JD requirement strings.
    
    Handles formats like:
    - (SQL OR SQL DDL OR SQL-Based Access Paths) AND (COBOL)
    - CompilersOR LinkersOR Debuggers (no space before OR)
    - Python or NumPy or Pandas
    """
    if not s or pd.isna(s):
        return []
    s = str(s)
    s = re.sub(r'\s*over\s+\d+\s*year\(s\)\s*', '', s, flags=re.IGNORECASE)
    groups = re.split(r'\)\s*AND\s*\(|\)\s*AND\(', s)
    skills = []
    for g in groups:
        g = g.strip('() ')
        # Split by OR — handle both "X OR Y" and "Xor Y" (no space before or)
        for p in re.split(r'\s*[Oo][Rr]\s+|\s+[Oo][Rr]\s*', g):
            sk = p.strip('() ').strip()
            if sk and 1 < len(sk) < 60:
                skip = ['YEARS', 'YEAR', 'SENIOR', 'LEAD', 'JUNIOR', 'BACHELOR', 'MASTER',
                        'DEGREE', 'STRONG', 'EXCELLENT', 'GOOD', 'EXPERIENCE', 'SKILLS']
                if not any(w in sk.upper() for w in skip):
                    skills.append(sk.title())
    return list(set(skills))


df['parsed_skills'] = df['skills_clean'].apply(parse_skills)

# Filter to last 12 months (relevant data)
cutoff = pd.Timestamp('2025-06-01')
recent = df[df['issue_date'] >= cutoff].copy()

print("=" * 70)
print("  BUILDING COMPANY-SPECIFIC SKILL PROFILES")
print("=" * 70)
print(f"  Total JDs (last 12m): {len(recent):,}")
print(f"  Unique companies: {recent['company_name'].nunique()}")
print(f"  Unique clusters: {recent['cluster_name'].nunique()}")

# ─── Build profiles: per company × per cluster ───
profiles = []
grouped = recent.groupby(['company_name', 'cluster_name'])

for (company, cluster), grp in grouped:
    jd_count = len(grp)
    total_openings = int(grp['openings'].sum())

    # Top skills for THIS company in THIS cluster
    all_skills = []
    for sl in grp['parsed_skills']:
        all_skills.extend(sl)
    skill_counts = pd.Series(all_skills).value_counts()
    top_skills = skill_counts.head(10).index.tolist()

    # Top locations for THIS company in THIS cluster
    loc_counts = grp['location'].value_counts()
    top_locations = loc_counts.head(5).index.tolist()

    profiles.append({
        'company_name': company,
        'cluster_name': cluster,
        'jd_count': jd_count,
        'openings': total_openings,
        'top_skills': json.dumps(top_skills),
        'top_locations': json.dumps(top_locations),
    })

profiles_df = pd.DataFrame(profiles)
print(f"  Company x Cluster profiles: {len(profiles_df):,}")

# ─── Create table and insert ───
conn = mysql.connector.connect(
    host='localhost', port=3305,
    user='resume_user', password='resume_password',
    database='resume_processing'
)
cur = conn.cursor()

cur.execute("DROP TABLE IF EXISTS company_cluster_profiles")
cur.execute("""
    CREATE TABLE company_cluster_profiles (
        id INT AUTO_INCREMENT PRIMARY KEY,
        company_name VARCHAR(200) NOT NULL,
        cluster_name VARCHAR(100) NOT NULL,
        jd_count INT NOT NULL,
        openings INT NOT NULL,
        top_skills JSON,
        top_locations JSON,
        INDEX idx_company (company_name),
        INDEX idx_cluster (cluster_name),
        INDEX idx_company_cluster (company_name, cluster_name)
    )
""")

insert_sql = """
    INSERT INTO company_cluster_profiles 
    (company_name, cluster_name, jd_count, openings, top_skills, top_locations)
    VALUES (%s, %s, %s, %s, %s, %s)
"""

rows_inserted = 0
for _, row in profiles_df.iterrows():
    cur.execute(insert_sql, (
        row['company_name'], row['cluster_name'],
        row['jd_count'], row['openings'],
        row['top_skills'], row['top_locations'],
    ))
    rows_inserted += 1

conn.commit()

# ─── Verify with examples ───
print(f"\n  Inserted {rows_inserted:,} profiles into company_cluster_profiles")

# Show the difference: cluster-level vs company-specific
print(f"\n{'=' * 70}")
print("  PROOF: Cluster-level vs Company-specific skills")
print(f"{'=' * 70}")

test_clusters = [
    'US | Engineering - General',
    'US | Software Engineer',
    'IN | Software Engineer',
]

for cluster in test_clusters:
    # Cluster-level skills (from demand_forecasts)
    cur.execute("SELECT top_skills FROM demand_forecasts WHERE cluster_name = %s LIMIT 1", (cluster,))
    row = cur.fetchone()
    cluster_skills = json.loads(row[0]) if row and row[0] else []

    # Company-specific skills
    cur.execute("""
        SELECT company_name, top_skills, jd_count 
        FROM company_cluster_profiles 
        WHERE cluster_name = %s 
        ORDER BY jd_count DESC LIMIT 5
    """, (cluster,))
    company_rows = cur.fetchall()

    print(f"\n  {cluster}")
    print(f"    Cluster-level skills: {cluster_skills[:6]}")
    for comp, skills, count in company_rows:
        cs = json.loads(skills)[:5]
        print(f"    {comp:<30} ({count:>3} JDs): {cs}")

# Show all unique companies
cur.execute("SELECT DISTINCT company_name FROM company_cluster_profiles ORDER BY company_name")
all_companies = [r[0] for r in cur.fetchall()]
print(f"\n  All companies with profiles ({len(all_companies)}):")
for c in all_companies[:30]:
    print(f"    {c}")
if len(all_companies) > 30:
    print(f"    ... and {len(all_companies) - 30} more")

cur.close()
conn.close()

print(f"\n{'=' * 70}")
print("  DONE. API can now serve company-specific skills.")
print(f"{'=' * 70}")
