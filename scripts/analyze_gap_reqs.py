"""
Clean analysis of '2026 Gap Reqs.xlsx' — the new company (Gap Inc.)
"""
import pandas as pd
import os

file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '2026 Gap Reqs.xlsx')

df = pd.read_excel(file_path, sheet_name='Sheet2')

# Clean up - parse dates
df['Created Date'] = pd.to_datetime(df['Created Date'], errors='coerce')
df['Release Date'] = pd.to_datetime(df['Release Date'], errors='coerce')
df['Fill Date'] = pd.to_datetime(df['Fill Date'], errors='coerce')

# Drop rows with no job title
df = df.dropna(subset=['Job Title'])
df = df[df['Job Title'].str.strip() != '']

print("=" * 70)
print("  2026 GAP REQS — NEW COMPANY DATA SUMMARY")
print("=" * 70)
print(f"\n  Total valid reqs: {len(df)}")
print(f"  Date range: {df['Created Date'].min()} → {df['Created Date'].max()}")
print(f"  Unique job titles: {df['Job Title'].nunique()}")
print(f"  Locations: {df['City'].nunique()} cities, {df['State Name'].nunique()} states")

# Monthly breakdown
df['month'] = df['Created Date'].dt.to_period('M')
monthly = df.groupby('month').size()
print(f"\n  MONTHLY BREAKDOWN:")
for month, count in monthly.items():
    print(f"    {month}: {count} reqs")

# Job title frequency
print(f"\n  TOP 30 JOB TITLES:")
title_counts = df['Job Title'].value_counts()
for title, count in title_counts.head(30).items():
    print(f"    {title:<55} {count}")

# Locations
print(f"\n  TOP LOCATIONS:")
loc_counts = df.groupby(['City', 'State Name']).size().sort_values(ascending=False)
for (city, state), count in loc_counts.head(15).items():
    print(f"    {city}, {state}: {count}")

# Rate card stats
print(f"\n  RATE CARD STATS:")
print(f"    Mean: ${df['Rate Card'].mean():.2f}/hr")
print(f"    Median: ${df['Rate Card'].median():.2f}/hr")
print(f"    Range: ${df['Rate Card'].min():.2f} - ${df['Rate Card'].max():.2f}/hr")

# Fill rate
filled = df['Fill Date'].notna().sum()
print(f"\n  FILL RATE: {filled}/{len(df)} = {filled/len(df)*100:.1f}%")

# Now try to map to our existing clusters
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
ttc = pd.read_pickle(os.path.join(data_dir, 'title_to_cluster.pkl'))
title_to_role = dict(zip(ttc['raw_title'], ttc['role_cluster']))

def classify(title):
    if title in title_to_role:
        return title_to_role[title]
    title_lower = title.lower().strip()
    for raw_t, cluster in title_to_role.items():
        if raw_t.lower() in title_lower or title_lower in raw_t.lower():
            return cluster
    return None

df['mapped_cluster'] = df['Job Title'].apply(classify)
mapped = df[df['mapped_cluster'].notna()]
unmapped = df[df['mapped_cluster'].isna()]

print(f"\n{'='*70}")
print(f"  CLUSTER MAPPING RESULTS (using existing title_to_cluster.pkl)")
print(f"{'='*70}")
print(f"  Mapped: {len(mapped)}/{len(df)} = {len(mapped)/len(df)*100:.1f}%")
print(f"  Unmapped: {len(unmapped)}/{len(df)} = {len(unmapped)/len(df)*100:.1f}%")

if not mapped.empty:
    print(f"\n  MAPPED TO CLUSTERS:")
    cluster_counts = mapped['mapped_cluster'].value_counts()
    for cluster, count in cluster_counts.items():
        print(f"    {cluster:<40} {count}")

if not unmapped.empty:
    print(f"\n  UNMAPPED TITLES ({len(unmapped)}):")
    for title in unmapped['Job Title'].unique():
        print(f"    ❌ {title}")
