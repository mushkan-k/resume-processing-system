"""Investigate why pickle has more data than JobDiva API for some months."""
import pandas as pd
import numpy as np
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_pickle(ROOT / 'data/clean_42k_v1.pkl')
df['month'] = pd.to_datetime(df['issue_date']).dt.to_period('M')

print('='*70)
print('  DATA SOURCE INVESTIGATION — WHY PICKLE > API?')
print('='*70)

# 1. Exact duplicates
print(f'\n--- 1. DUPLICATE CHECK ---')
dups_exact = df.duplicated()
print(f'  Exact duplicate rows: {dups_exact.sum()} / {len(df)}')

key_cols = ['issue_date', 'title', 'company_name', 'openings']
dups_key = df.duplicated(subset=key_cols)
print(f'  Dup on (date+title+company+openings): {dups_key.sum()}')

key_cols2 = ['issue_date', 'title', 'company_name']
dups_key2 = df.duplicated(subset=key_cols2, keep=False)
print(f'  Same (date+title+company) appearing multiple times: {dups_key2.sum()} rows')

# 2. Jan 2026 deep dive (pickle=774, API=417)
print(f'\n--- 2. JANUARY 2026 DEEP DIVE (Pickle=774, API=417) ---')
jan = df[df['month'] == pd.Period('2026-01')]
print(f'  Rows: {len(jan)}, Openings: {jan["openings"].sum()}')
print(f'  Unique companies: {jan["company_name"].nunique()}')
print(f'  Unique titles: {jan["title"].nunique()}')
dates = pd.to_datetime(jan['issue_date']).dt.date
print(f'  Unique dates: {dates.nunique()}')
print(f'  Date range: {dates.min()} to {dates.max()}')

# Same job appearing multiple times?
jan_grp = jan.groupby(['issue_date', 'title', 'company_name']).agg(
    count=('openings', 'size'), total_openings=('openings', 'sum')
).reset_index()
multi = jan_grp[jan_grp['count'] > 1]
print(f'\n  Same (date, title, company) appearing >1 time: {len(multi)} cases')
if len(multi) > 0:
    extra_rows = multi['count'].sum() - len(multi)
    extra_openings = jan_grp[jan_grp['count'] > 1].merge(
        jan, on=['issue_date','title','company_name'])['openings'].sum()
    print(f'  These create {extra_rows} extra rows')
    print(f'\n  Top 10 examples:')
    for _, r in multi.nlargest(10, 'count').iterrows():
        print(f'    {str(r["issue_date"])[:10]} | {r["company_name"]:<20} | {r["title"][:35]:<35} | x{r["count"]} ({r["total_openings"]} openings)')

# 3. Check if duplicates differ in region/location
print(f'\n--- 3. DO DUPLICATES HAVE DIFFERENT REGIONS? ---')
# Take a sample multi-row case
if len(multi) > 0:
    sample = multi.iloc[0]
    sample_rows = jan[(jan['issue_date'] == sample['issue_date']) & 
                       (jan['title'] == sample['title']) & 
                       (jan['company_name'] == sample['company_name'])]
    print(f'  Sample: {sample["company_name"]} / {sample["title"][:40]}')
    print(f'  Rows:')
    for _, r in sample_rows.iterrows():
        print(f'    region={r["region"]:<6} location={r["location"]:<30} openings={r["openings"]}')

# 4. Overall: how many rows are "real duplicates" across entire dataset?
print(f'\n--- 4. FULL DATASET DUPLICATE ANALYSIS ---')
all_grp = df.groupby(['issue_date', 'title', 'company_name']).size().reset_index(name='count')
single = (all_grp['count'] == 1).sum()
multi_all = (all_grp['count'] > 1).sum()
total_extra = all_grp[all_grp['count'] > 1]['count'].sum() - multi_all
print(f'  Unique job records (1 row): {single:,}')
print(f'  Multi-row job records (>1 row): {multi_all:,}')
print(f'  Extra rows from multiples: {total_extra:,}')
print(f'  If deduplicated, dataset would be: {len(df) - total_extra:,} rows (currently {len(df):,})')
print(f'  Extra data: {total_extra/len(df)*100:.1f}% of rows are duplicates')

# 5. Are duplicates same region or different regions?
print(f'\n--- 5. DUPLICATE PATTERN: SAME vs DIFFERENT REGIONS ---')
multi_keys = all_grp[all_grp['count'] > 1][['issue_date', 'title', 'company_name']]
multi_rows = df.merge(multi_keys, on=['issue_date', 'title', 'company_name'])
# For each group, check if regions differ
region_check = multi_rows.groupby(['issue_date', 'title', 'company_name'])['region'].nunique()
same_region = (region_check == 1).sum()
diff_region = (region_check > 1).sum()
print(f'  Multi-rows with SAME region: {same_region} (true duplicates)')
print(f'  Multi-rows with DIFFERENT region: {diff_region} (legit — same job posted in multiple regions)')

# 6. Recommendation
print(f'\n{"="*70}')
print(f'  RECOMMENDATION')
print(f'{"="*70}')
if same_region > 0:
    print(f'  ⚠️  {same_region} job records are TRUE DUPLICATES (same region)')
    print(f'     These should be deduplicated — they add noise without signal')
if diff_region > 0:
    print(f'  ✅ {diff_region} multi-row records are legit (different regions)')
    print(f'     These represent same job posted across regions — keep them')
