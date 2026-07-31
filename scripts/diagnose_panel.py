"""Quick check: does the zero-filled panel extend to June 2026 for all combos?
And what are the last lag values for Developer combos?"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_pickle('data/clean_42k_v1.pkl')
ttc = pd.read_pickle('data/title_to_cluster.pkl')
title_map = dict(zip(ttc['raw_title'], ttc['role_cluster']))
df['role_cluster'] = df['title'].map(title_map)
df = df.dropna(subset=['role_cluster'])
df['month'] = pd.to_datetime(df['issue_date']).dt.to_period('M')

# Drop incomplete July
monthly_total = df.groupby('month')['openings'].sum().sort_index()
df = df[df['month'] != monthly_total.index[-1]]

# Build panel same way as generate script
ts = df.groupby(['role_cluster', 'company_name', 'month']).agg(openings=('openings', 'sum')).reset_index()
combo_months = ts.groupby(['role_cluster', 'company_name'])['month'].nunique()
viable = combo_months[combo_months >= 20].index.tolist()
ts['combo'] = list(zip(ts['role_cluster'], ts['company_name']))
panel = ts[ts['combo'].isin(viable)].copy()

# Zero-fill
filled_parts = []
for combo_key, grp in panel.groupby('combo'):
    grp_idx = grp.set_index('month')['openings']
    combo_months_range = pd.period_range(grp['month'].min(), grp['month'].max(), freq='M')
    filled = grp_idx.reindex(combo_months_range, fill_value=0).reset_index()
    filled.columns = ['month', 'openings']
    filled['combo'] = [combo_key] * len(filled)
    filled['role_cluster'] = combo_key[0]
    filled['company_name'] = combo_key[1]
    filled_parts.append(filled)
panel = pd.concat(filled_parts, ignore_index=True)

# Lags
panel = panel.sort_values(['role_cluster', 'company_name', 'month']).reset_index(drop=True)
panel['lag1'] = panel.groupby('combo')['openings'].shift(1)
panel['lag2'] = panel.groupby('combo')['openings'].shift(2)
panel['lag3'] = panel.groupby('combo')['openings'].shift(3)
panel['roll3'] = panel.groupby('combo')['openings'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
panel = panel.dropna(subset=['lag1', 'lag2', 'lag3', 'roll3'])

# Check Developer combos - what's their last month and lag values?
dev = panel[panel['role_cluster'] == 'Developer']
print("=== DEVELOPER COMBOS: LAST ROW IN PANEL ===")
last_rows = dev.groupby('company_name').last()
for co in last_rows.index:
    r = last_rows.loc[co]
    print(f"  {co:25s} last_month={r['month']} openings={r['openings']:.0f} lag1={r['lag1']:.0f} lag2={r['lag2']:.0f} lag3={r['lag3']:.0f}")

# Key question: do some combos end BEFORE June 2026?
print(f"\n=== COMBO END MONTHS ===")
last_months = panel.groupby('combo')['month'].max()
print(f"  Combos ending at June 2026: {(last_months == '2026-06').sum()}")
print(f"  Combos ending BEFORE June 2026: {(last_months < '2026-06').sum()}")
print(f"  Earliest end: {last_months.min()}")
print(f"\n  Combos ending before 2025:")
old_combos = last_months[last_months < '2025-01']
print(f"    Count: {len(old_combos)}")
if len(old_combos) > 0:
    for combo, month in old_combos.sort_values().head(10).items():
        print(f"    {combo[0]:25s} | {combo[1]:20s} | ends {month}")
