import pandas as pd
from collections import Counter

t2c = pd.read_pickle(r"c:\Users\serveradmin\Desktop\resume-processing-system\data\title_to_cluster.pkl")

# Big clusters to analyze
big_clusters = ['Engineering - General', 'ERP Consultant', 'Specialist - General', 
                'Developer', 'Data Analyst', 'Product Manager', 'Program Manager',
                'Project Manager', 'Operations - General', 'Associate - General']

# Load actual JD data to see which titles have volume
df = pd.read_pickle(r"c:\Users\serveradmin\Desktop\resume-processing-system\data\clean_42k_v1.pkl")
df['issue_date'] = pd.to_datetime(df['issue_date'])
df_2026 = df[df['issue_date'] >= '2026-01-01']

title_map = dict(zip(t2c['raw_title'].str.lower().str.strip(), t2c['role_cluster']))
df_2026 = df_2026.copy()
df_2026['cluster'] = df_2026['title'].str.lower().str.strip().map(title_map)

print("=" * 80)
print("BIG CLUSTER BREAKDOWN — Job Titles with JD counts in 2026")
print("=" * 80)

for cluster in big_clusters:
    cdf = df_2026[df_2026['cluster'] == cluster]
    if len(cdf) == 0:
        continue
    
    title_counts = cdf.groupby('title').agg(
        jds=('title', 'count'),
        openings=('openings', 'sum')
    ).sort_values('jds', ascending=False)
    
    print(f"\n{'─'*80}")
    print(f"{cluster} — {len(cdf)} JDs, {len(title_counts)} unique titles")
    print(f"{'─'*80}")
    print(f"  {'Title':<60} {'JDs':>5} {'Opens':>6}")
    
    for title, row in title_counts.head(30).iterrows():
        print(f"  {title[:60]:<60} {row['jds']:>5} {row['openings']:>6}")
    
    if len(title_counts) > 30:
        remaining = title_counts.iloc[30:]
        print(f"  ... +{len(remaining)} more titles ({remaining['jds'].sum()} JDs)")
