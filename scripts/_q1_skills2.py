import pandas as pd, os, json
from collections import Counter

DATA = 'c:/Users/serveradmin/Desktop/resume-processing-system/data'
df = pd.read_pickle(os.path.join(DATA, 'clean_42k_v1.pkl'))
ttc = pd.read_pickle(os.path.join(DATA, 'title_to_cluster.pkl'))
df['role_cluster'] = df['title'].map(dict(zip(ttc['raw_title'], ttc['role_cluster'])))
df = df.dropna(subset=['role_cluster'])
df['issue_date'] = pd.to_datetime(df['issue_date'])
if 'region' not in df.columns:
    df['region'] = 'US'
df['region'] = df['region'].fillna('US')

eng = df[(df['region'] == 'US') & (df['role_cluster'] == 'Engineering - General')]
q1 = eng[(eng['issue_date'] >= '2026-01-01') & (eng['issue_date'] <= '2026-03-31')]

print(f"US | Engineering - General — Q1 2026")
print(f"  JDs: {len(q1):,}  |  Openings: {q1['openings'].sum():,}  |  Titles: {q1['title'].nunique()}")

# Sample skills_clean
sample = q1['skills_clean'].dropna().head(3).tolist()
print(f"\n  Sample skills_clean type: {type(sample[0]) if sample else 'N/A'}")
for i, s in enumerate(sample[:3]):
    print(f"  [{i}] {str(s)[:150]}")

# Parse skills
all_skills = Counter()
jds_with_skills = 0
for val in q1['skills_clean'].dropna():
    if isinstance(val, list):
        skills = val
    elif isinstance(val, str):
        try:
            skills = json.loads(val)
        except:
            skills = [s.strip() for s in val.split(',') if s.strip()]
    else:
        continue
    if skills:
        jds_with_skills += 1
        for s in skills:
            s = str(s).strip()
            if s and len(s) > 1:
                all_skills[s] += 1

print(f"\n  JDs with skills: {jds_with_skills} / {len(q1)}")
print(f"  Unique skills: {len(all_skills)}")
print(f"  Total mentions: {sum(all_skills.values())}")

# Filter junk
SKIP = {'skills to be assigned', 'null', 'n/a', 'none', 'tbd'}
clean = {k: v for k, v in all_skills.items() 
         if k.lower() not in SKIP and '{' not in k and len(k) < 50}

print(f"  After filtering: {len(clean)} unique skills")
print(f"\n  Top 40 skills (Q1 2026, 242 JDs):")
print(f"  {'Rank':>4}  {'Count':>5}  {'% of JDs':>8}  Skill")
print(f"  {'-'*60}")
for i, (skill, count) in enumerate(sorted(clean.items(), key=lambda x: -x[1])[:40], 1):
    pct = count / jds_with_skills * 100
    print(f"  {i:4d}  {count:5d}  {pct:7.1f}%  {skill}")
