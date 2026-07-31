import pandas as pd, os
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

# Filter: US | Engineering - General, Q1 2026
eng = df[(df['region'] == 'US') & (df['role_cluster'] == 'Engineering - General')]
q1 = eng[(eng['issue_date'] >= '2026-01-01') & (eng['issue_date'] <= '2026-03-31')]

print(f"US | Engineering - General — Q1 2026")
print(f"  JDs: {len(q1):,}")
print(f"  Openings: {q1['openings'].sum():,}")
print(f"  Unique titles: {q1['title'].nunique()}")

# Check what skill columns we have
skill_cols = [c for c in q1.columns if 'skill' in c.lower()]
print(f"\n  Skill columns available: {skill_cols}")

# Extract skills from the 'skills' or 'jd_skills' column
for col in ['skills', 'jd_skills', 'extracted_skills', 'skill']:
    if col in q1.columns:
        print(f"\n  Using column: '{col}'")
        sample = q1[col].dropna().head(3).tolist()
        print(f"  Sample values: {sample[:2]}")
        break

# Try parsing skills
all_skills = Counter()
skill_col = None
for col in ['skills', 'jd_skills', 'extracted_skills', 'skill']:
    if col in q1.columns:
        skill_col = col
        break

if skill_col:
    for val in q1[skill_col].dropna():
        if isinstance(val, list):
            for s in val:
                all_skills[s.strip()] += 1
        elif isinstance(val, str):
            # Try comma-separated or JSON
            import json
            try:
                parsed = json.loads(val)
                if isinstance(parsed, list):
                    for s in parsed:
                        all_skills[str(s).strip()] += 1
                    continue
            except:
                pass
            for s in val.split(','):
                s = s.strip()
                if s:
                    all_skills[s] += 1

    print(f"\n  Total unique skills extracted: {len(all_skills)}")
    print(f"  Total skill mentions: {sum(all_skills.values())}")
    print(f"\n  Top 30 skills:")
    for skill, count in all_skills.most_common(30):
        print(f"    {count:5d}  {skill[:60]}")
else:
    print("\n  No skill column found! Columns:", list(q1.columns))

# Also check if 'description' or 'jd_text' exists for NLP extraction
text_cols = [c for c in q1.columns if any(x in c.lower() for x in ['desc', 'jd_text', 'body', 'full_text'])]
print(f"\n  Text columns for NLP: {text_cols}")
has_text = 0
for col in text_cols:
    has_text = q1[col].dropna().shape[0]
    print(f"    {col}: {has_text} non-null")
