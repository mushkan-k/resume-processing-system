"""
Panel Cleaning Pass — Data Quality Audit
==========================================
Checks: date range, missing vs zero, outliers, duplicate companies
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_pickle(ROOT / 'data/clean_42k_v1.pkl')

# Normalize column names — raw data uses issue_date, panel uses month
if 'month' not in df.columns and 'issue_date' in df.columns:
    df['month'] = pd.to_datetime(df['issue_date']).dt.to_period('M')
if 'role_cluster' not in df.columns and 'title' in df.columns:
    # Load title_to_cluster mapping (DataFrame with raw_title → role_cluster)
    import pickle
    ttc = pd.read_pickle(ROOT / 'data/title_to_cluster.pkl')
    title_map = dict(zip(ttc['raw_title'], ttc['role_cluster']))
    df['role_cluster'] = df['title'].map(title_map)
    df = df.dropna(subset=['role_cluster'])

print('=' * 70)
print('  PANEL CLEANING PASS — DATA QUALITY AUDIT')
print('=' * 70)

# 1. Historical depth
print(f'\n{"─"*70}')
print(f'  1. HISTORICAL DEPTH')
print(f'{"─"*70}')
print(f'  Columns: {list(df.columns)}')
print(f'  Shape: {df.shape}')
print(f'  Date range: {df["month"].min()} to {df["month"].max()}')
print(f'  Unique months: {df["month"].nunique()}')
n_months = df["month"].nunique()
print(f'  Span: ~{n_months} months ({n_months/12:.1f} years)')
print(f'  Unique combos: {df.groupby(["company_name","role_cluster"]).ngroups}')

# 2. Missing vs Zero
print(f'\n{"─"*70}')
print(f'  2. MISSING vs ZERO CHECK')
print(f'{"─"*70}')
nulls = df.isnull().sum()
print(f'  Null counts:')
for col, n in nulls.items():
    if n > 0:
        print(f'    {col}: {n} nulls ({n/len(df)*100:.1f}%)')
if nulls.sum() == 0:
    print(f'    No nulls anywhere ✅')

zero_openings = (df['openings'] == 0).sum()
print(f'\n  Zero openings: {zero_openings} / {len(df)} ({zero_openings/len(df)*100:.1f}%)')

# Check if zeros are "real zeros" or "missing data coded as zero"
# Look at combos that have ONLY zeros in certain stretches
combo_zero_pct = df.groupby(['company_name', 'role_cluster']).apply(
    lambda g: (g['openings'] == 0).mean()).reset_index(name='zero_pct')
always_zero = combo_zero_pct[combo_zero_pct['zero_pct'] == 1.0]
mostly_zero = combo_zero_pct[combo_zero_pct['zero_pct'] > 0.8]
print(f'  Combos that are 100% zero: {len(always_zero)}')
print(f'  Combos that are >80% zero: {len(mostly_zero)}')

# Consecutive zeros (suspicious gaps)
def max_consecutive_zeros(series):
    max_run = 0
    current = 0
    for v in series:
        if v == 0:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run

consec = df.sort_values('month').groupby(['company_name', 'role_cluster'])['openings'].apply(max_consecutive_zeros)
long_gaps = consec[consec >= 6]  # 6+ consecutive months of zero
print(f'  Combos with 6+ consecutive zero months: {len(long_gaps)}')
if len(long_gaps) > 0:
    print(f'  (These might be "data not available" rather than "zero demand")')
    top_gaps = long_gaps.nlargest(10)
    for (comp, role), gap in top_gaps.items():
        print(f'    {comp:<20} {role:<30} {gap} months of zeros')

# 3. Outlier scan
print(f'\n{"─"*70}')
print(f'  3. OUTLIER SCAN')
print(f'{"─"*70}')
print(f'  Openings distribution:')
print(f'    mean={df["openings"].mean():.1f}, median={df["openings"].median():.1f}, '
      f'std={df["openings"].std():.1f}, max={df["openings"].max()}')
for pct in [90, 95, 99]:
    print(f'    {pct}th percentile: {df["openings"].quantile(pct/100):.0f}')

# IQR-based outliers
Q1 = df['openings'].quantile(0.25)
Q3 = df['openings'].quantile(0.75)
IQR = Q3 - Q1
upper_fence = Q3 + 3 * IQR  # Using 3x IQR (extreme outliers)
outliers = df[df['openings'] > upper_fence]
print(f'\n  IQR upper fence (Q3 + 3*IQR): {upper_fence:.0f}')
print(f'  Extreme outliers (>{upper_fence:.0f}): {len(outliers)} rows ({len(outliers)/len(df)*100:.2f}%)')
if len(outliers) > 0:
    print(f'\n  Top 15 outlier rows:')
    top = outliers.nlargest(15, 'openings')[['company_name', 'role_cluster', 'month', 'openings']]
    print(f'    {"Company":<22} {"Role Cluster":<30} {"Month":<10} {"Openings":>8}')
    print(f'    {"─"*72}')
    for _, r in top.iterrows():
        print(f'    {r["company_name"]:<22} {r["role_cluster"]:<30} {str(r["month"]):<10} {r["openings"]:>8}')

# 4. Duplicate company names
print(f'\n{"─"*70}')
print(f'  4. COMPANY NAME DEDUPLICATION CHECK')
print(f'{"─"*70}')
companies = sorted(df['company_name'].unique())
print(f'  Total unique companies: {len(companies)}')

# Check for "IN " prefix pattern (India vs non-India?)
in_companies = [c for c in companies if c.startswith('IN ')]
non_in = [c for c in companies if not c.startswith('IN ')]
print(f'\n  Companies with "IN " prefix: {len(in_companies)}')
overlaps = []
for c in in_companies:
    base = c[3:]
    if base in non_in:
        overlaps.append((c, base))
        vol_in = df[df['company_name'] == c]['openings'].sum()
        vol_base = df[df['company_name'] == base]['openings'].sum()
        print(f'    ⚠️  "{c}" ({vol_in:.0f} openings) AND "{base}" ({vol_base:.0f} openings) BOTH EXIST')
    else:
        vol = df[df['company_name'] == c]['openings'].sum()
        print(f'    "{c}" ({vol:.0f} openings) — no non-IN counterpart')

# Fuzzy near-duplicates
prefix_groups = defaultdict(list)
for c in companies:
    key = c.lower().replace(' ', '').replace('-', '').replace('_', '')[:8]
    prefix_groups[key].append(c)
dups = {k: v for k, v in prefix_groups.items() if len(v) > 1}
if dups:
    print(f'\n  Other potential duplicates (similar names):')
    for prefix, names in sorted(dups.items()):
        if not any(n.startswith('IN ') for n in names):  # Skip IN pairs already shown
            print(f'    {names}')

# 5. Combo density summary
print(f'\n{"─"*70}')
print(f'  5. COMBO DENSITY SUMMARY')
print(f'{"─"*70}')
combo_sizes = df.groupby(['company_name', 'role_cluster']).size()
print(f'  Rows per combo: mean={combo_sizes.mean():.1f}, median={combo_sizes.median():.0f}, '
      f'min={combo_sizes.min()}, max={combo_sizes.max()}')
print(f'  Combos with <6 months data: {(combo_sizes < 6).sum()}')
print(f'  Combos with <12 months data: {(combo_sizes < 12).sum()}')
print(f'  Combos with full history ({n_months} months): {(combo_sizes == n_months).sum()}')

# 6. Actionable recommendations
print(f'\n{"═"*70}')
print(f'  RECOMMENDATIONS')
print(f'{"═"*70}')
if len(overlaps) > 0:
    print(f'  🔧 MERGE "IN X" + "X" companies? ({len(overlaps)} pairs found)')
    print(f'     → Would reduce combo count, increase data density per combo')
if len(long_gaps) > 0:
    print(f'  🔧 INVESTIGATE {len(long_gaps)} combos with 6+ month zero-gaps')
    print(f'     → May be missing data, not actual zero demand')
if len(outliers) > 0:
    print(f'  🔧 CAP/WINSORIZE {len(outliers)} extreme outlier rows (>{upper_fence:.0f} openings)')
    print(f'     → Or investigate if data entry errors')
print(f'  📊 HISTORICAL DEPTH: {n_months} months — check if JobDiva has more')
print(f'     → More months = better trend estimation for Bayesian model')
