"""Diagnose WHY month-1 forecast is already 10-20x too high.
The issue: Developer actual is ~40/month, forecast is 873 for July.
Is this a per-combo problem or an aggregation problem?"""
import pandas as pd
import numpy as np

df = pd.read_pickle('data/clean_42k_v1.pkl')
ttc = pd.read_pickle('data/title_to_cluster.pkl')
title_map = dict(zip(ttc['raw_title'], ttc['role_cluster']))
df['role_cluster'] = df['title'].map(title_map)
df = df.dropna(subset=['role_cluster'])
df['month'] = pd.to_datetime(df['issue_date']).dt.to_period('M')

# How many combos are in "Developer"?
dev = df[df['role_cluster'] == 'Developer']
dev_ts = dev.groupby(['company_name', 'month']).agg(openings=('openings', 'sum')).reset_index()
combo_months = dev_ts.groupby('company_name')['month'].nunique()
viable_devs = combo_months[combo_months >= 20]

print(f"Developer cluster:")
print(f"  Total companies: {dev['company_name'].nunique()}")
print(f"  Viable companies (>=20 months): {len(viable_devs)}")
print(f"  Last 6 months actual total: {dev[dev['month'] >= '2026-01']['openings'].sum()}")
print(f"  Last 6 months monthly avg: {dev[dev['month'] >= '2026-01'].groupby('month')['openings'].sum().mean():.0f}")

# Per-company monthly avg for Developer
print(f"\n  Viable Developer combos (last 6 months avg):")
for co in viable_devs.index[:10]:
    recent = dev_ts[(dev_ts['company_name'] == co) & (dev_ts['month'] >= '2026-01')]
    avg = recent['openings'].mean() if len(recent) > 0 else 0
    total_months = combo_months[co]
    print(f"    {co:30s}: avg={avg:.1f}/month, {total_months} months history")

# The key question: after zero-fill, what's the MEAN opening for Developer combos?
# The model trains on log1p(openings). With lots of zeros, log1p(0)=0.
# But the last few months have NON-zero values.
# The issue might be: the model's intercept + combo effects = high baseline,
# and lags of real recent values (not zeros) push predictions up.

# Let's check: what are the lag features for the last row of each Developer combo?
print(f"\n=== DEVELOPER COMBO LAST-KNOWN LAGS ===")
ts_all = df.groupby(['role_cluster', 'company_name', 'month']).agg(openings=('openings', 'sum')).reset_index()
# Zero-fill within span
dev_viable_data = ts_all[(ts_all['role_cluster'] == 'Developer') & 
                          (ts_all['company_name'].isin(viable_devs.index))]
print(f"  # viable combos for Developer: {len(viable_devs)}")
print(f"  Sum of LAST known openings across all Developer combos:")

# What would the model see as lag1 for each combo?
last_vals = dev_viable_data.sort_values('month').groupby('company_name')['openings'].last()
print(f"  Sum of last openings: {last_vals.sum()}")
print(f"  This is what drives month-1 forecast when summed across {len(last_vals)} combos")
print(f"  But actual Developer total in recent months is only ~40/month")
print(f"\n  TOP 5 last-opening values:")
for co, v in last_vals.sort_values(ascending=False).head(5).items():
    print(f"    {co}: {v}")
