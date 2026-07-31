import pandas as pd, os
DATA = 'c:/Users/serveradmin/Desktop/resume-processing-system/data'
df = pd.read_pickle(os.path.join(DATA, 'clean_42k_v1.pkl'))
ttc = pd.read_pickle(os.path.join(DATA, 'title_to_cluster.pkl'))
df['role_cluster'] = df['title'].map(dict(zip(ttc['raw_title'], ttc['role_cluster'])))
df = df.dropna(subset=['role_cluster'])
if 'region' not in df.columns:
    df['region'] = 'US'
df['region'] = df['region'].fillna('US')
df['region'] = df['region'].apply(lambda r: r if r in ('US', 'IN') else 'LATAM & Others')

# Filter US | Engineering - General
eng = df[(df['region'] == 'US') & (df['role_cluster'] == 'Engineering - General')]
print(f"US | Engineering - General")
print(f"Total JDs: {len(eng):,}")
print(f"Total openings: {eng['openings'].sum():,}")
print(f"Unique titles: {eng['title'].nunique()}")
print(f"\n{'#':>4}  {'Title':50s}  {'JDs':>6}  {'Openings':>8}")
print("-" * 75)

title_stats = eng.groupby('title').agg(
    jds=('title', 'count'),
    openings=('openings', 'sum')
).sort_values('jds', ascending=False)

for i, (title, row) in enumerate(title_stats.iterrows(), 1):
    print(f"{i:4d}  {title[:50]:50s}  {row['jds']:6d}  {row['openings']:8d}")
