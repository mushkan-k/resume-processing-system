"""Check June 2026 data completeness in pickle."""
import pandas as pd

# Check pickle — this is the actual training data
df = pd.read_pickle('data/clean_42k_v1.pkl')
df['month'] = pd.to_datetime(df['issue_date']).dt.to_period('M')
mt = df.groupby('month').agg(row_count=('openings', 'count'), total=('openings', 'sum'))
print("=" * 50)
print("  Pickle (clean_42k_v1.pkl) — last 8 months")
print("=" * 50)
print(f"  {'Month':<10} {'Rows':>6} {'Openings':>10}")
print(f"  {'-'*28}")
for m, r in mt.tail(8).iterrows():
    flag = ' <-- LOW' if r['total'] < 300 else ''
    print(f"  {str(m):<10} {int(r['row_count']):>6} {int(r['total']):>10,}{flag}")

# 3. Check if June has daily distribution (is it cut off mid-month?)
jun = df[df['month'] == pd.Period('2026-06')]
if len(jun) > 0:
    jun_dates = pd.to_datetime(jun['issue_date'])
    print(f"\n  June 2026 date range: {jun_dates.min().date()} to {jun_dates.max().date()}")
    print(f"  Days with data: {jun_dates.dt.day.nunique()}")
    print(f"  Max day: {jun_dates.dt.day.max()}")
else:
    print("\n  No June 2026 data in pickle")
