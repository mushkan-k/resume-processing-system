"""Diagnose: which baseline predictor works best for H1 2026?"""
import pandas as pd, numpy as np, os

df = pd.read_pickle('data/clean_42k_v1.pkl')
ttc = pd.read_pickle('data/title_to_cluster.pkl')
df['role_cluster'] = df['title'].map(dict(zip(ttc['raw_title'], ttc['role_cluster'])))
df = df.dropna(subset=['role_cluster'])
df['issue_date'] = pd.to_datetime(df['issue_date'])
df['month'] = df['issue_date'].dt.to_period('M')
if 'region' not in df.columns: df['region'] = 'US'
df['region'] = df['region'].fillna('US')
df['region'] = df['region'].apply(lambda r: r if r in ('US', 'IN') else 'LATAM & Others')
df['region_role'] = df['region'] + ' | ' + df['role_cluster']

ts = df.groupby(['region_role', 'month'])['openings'].sum().reset_index()
ts['cal_month'] = ts['month'].dt.month
ts['year'] = ts['month'].dt.year

results = []
for rr in ts['region_role'].unique():
    sub = ts[ts['region_role'] == rr].sort_values('month')
    for m in range(1, 7):
        actual_row = sub[(sub['year'] == 2026) & (sub['cal_month'] == m)]
        if len(actual_row) == 0:
            continue
        actual = actual_row['openings'].values[0]

        smly_row = sub[(sub['year'] == 2025) & (sub['cal_month'] == m)]
        smly = smly_row['openings'].values[0] if len(smly_row) > 0 else None

        dec_row = sub[(sub['year'] == 2025) & (sub['cal_month'] == 12)]
        lag = dec_row['openings'].values[0] if len(dec_row) > 0 else None

        yr2025 = sub[sub['year'] == 2025]['openings']
        avg12 = yr2025.mean() if len(yr2025) > 0 else None

        if smly and lag and avg12 and actual > 0:
            results.append({
                'rr': rr, 'month': m, 'actual': actual,
                'smly': smly, 'lag': lag, 'avg12': avg12,
                'smly_err': abs(smly - actual) / actual,
                'lag_err': abs(lag - actual) / actual,
                'avg12_err': abs(avg12 - actual) / actual,
            })

rdf = pd.DataFrame(results)
print("Which predictor is best for H1 2026? (lower APE = better)\n")
print(f"  Same-month-last-year (SMLY):  median APE = {rdf['smly_err'].median()*100:.1f}%")
print(f"  Dec 2025 lag:                 median APE = {rdf['lag_err'].median()*100:.1f}%")
print(f"  12-month avg (2025):          median APE = {rdf['avg12_err'].median()*100:.1f}%")
print()
print(f"  SMLY wins vs lag: {(rdf['smly_err'] < rdf['lag_err']).mean()*100:.0f}% of the time")
print(f"  Avg12 wins vs lag: {(rdf['avg12_err'] < rdf['lag_err']).mean()*100:.0f}% of the time")
print()

# Total level
print("Portfolio level (sum across all clusters):")
total_actual = rdf['actual'].sum()
total_smly = rdf['smly'].sum()
total_lag = rdf['lag'].sum() * 6  # lag is same dec value repeated
total_avg12 = rdf['avg12'].sum()
print(f"  Actual H1 2026:   {total_actual:,.0f}")
print(f"  SMLY total:       {total_smly:,.0f}  ({total_smly/total_actual:.2f}x)")
print(f"  Avg12 total:      {total_avg12:,.0f}  ({total_avg12/total_actual:.2f}x)")
print()

# The fix: use SMLY as an anchor feature
print("==> The model should ANCHOR on same-month-last-year, not recent lags.")
print("    Lags carry Q4 dip forward. SMLY naturally adjusts for seasonality.")
