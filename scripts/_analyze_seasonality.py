"""
Analyze quarterly seasonality patterns from our actual data.
Check: Is Q1/Q3 high, Q2 slow, Q4 dip? Or something else?
"""
import pickle
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"

# Load training data
df = pickle.load(open(DATA_DIR / "clean_42k_v1.pkl", "rb"))
df['issue_date'] = pd.to_datetime(df['issue_date'])
df['year'] = df['issue_date'].dt.year
df['month'] = df['issue_date'].dt.month
df['quarter'] = df['issue_date'].dt.quarter

# Filter to meaningful years
df = df[df['year'] >= 2020]

print("=" * 70)
print("QUARTERLY SEASONALITY ANALYSIS (2020-2026)")
print("=" * 70)

# Overall quarterly pattern
print("\n1. TOTAL OPENINGS BY QUARTER (all years, all clusters):")
q_totals = df.groupby('quarter')['openings'].sum()
q_avg = q_totals / df.groupby('quarter')['year'].nunique()
total_avg = q_avg.mean()
print(f"{'Quarter':<10} {'Total':>8} {'Avg/Year':>10} {'Index':>8}")
print("-" * 40)
for q in [1, 2, 3, 4]:
    idx = q_avg[q] / total_avg
    print(f"Q{q:<9} {int(q_totals[q]):>8} {int(q_avg[q]):>10} {idx:>7.2f}x")

# Year-by-year quarterly breakdown
print("\n2. OPENINGS BY QUARTER × YEAR:")
pivot = df.groupby(['year', 'quarter'])['openings'].sum().unstack(fill_value=0)
print(pivot.to_string())

# Seasonal index per quarter (recent years: 2023-2025)
print("\n3. SEASONAL INDEX (2023-2025 only — most relevant):")
recent = df[(df['year'] >= 2023) & (df['year'] <= 2025)]
rq = recent.groupby('quarter')['openings'].sum()
rq_avg = rq / recent.groupby('quarter')['year'].nunique()
r_total_avg = rq_avg.mean()
print(f"{'Quarter':<10} {'Avg/Year':>10} {'Index':>8}")
print("-" * 30)
for q in [1, 2, 3, 4]:
    if q in rq_avg.index:
        idx = rq_avg[q] / r_total_avg
        print(f"Q{q:<9} {int(rq_avg[q]):>10} {idx:>7.2f}x")

# Monthly pattern
print("\n4. MONTHLY SEASONALITY (2023-2025):")
rm = recent.groupby('month')['openings'].sum()
rm_avg = rm / recent.groupby('month')['year'].nunique()
m_total_avg = rm_avg.mean()
print(f"{'Month':<8} {'Avg/Year':>10} {'Index':>8}")
print("-" * 30)
for m in range(1, 13):
    if m in rm_avg.index:
        idx = rm_avg[m] / m_total_avg
        print(f"{m:<8} {int(rm_avg[m]):>10} {idx:>7.2f}x")

# 2026 actuals pattern
print("\n5. 2026 ACTUALS (what we have):")
a2026 = df[df['year'] == 2026]
m2026 = a2026.groupby('month')['openings'].sum()
print(f"{'Month':<8} {'Openings':>10}")
print("-" * 20)
for m in sorted(m2026.index):
    print(f"{m:<8} {int(m2026[m]):>10}")
if len(m2026) > 0:
    print(f"{'Total':<8} {int(m2026.sum()):>10}")
    print(f"{'Avg/mo':<8} {int(m2026.mean()):>10}")

# Top clusters quarterly pattern
print("\n6. TOP 5 CLUSTERS — QUARTERLY PATTERN (2023-2025):")
top_clusters = recent.groupby('region_role' if 'region_role' in recent.columns else 'role_cluster')['openings'].sum().nlargest(5).index.tolist()
col = 'region_role' if 'region_role' in recent.columns else 'role_cluster'
for cluster in top_clusters:
    cdf = recent[recent[col] == cluster]
    cq = cdf.groupby('quarter')['openings'].sum()
    cq_avg = cq / cdf.groupby('quarter')['year'].nunique()
    c_mean = cq_avg.mean() if len(cq_avg) > 0 else 1
    indices = [f"Q{q}={cq_avg.get(q,0)/c_mean:.2f}x" for q in [1,2,3,4]]
    print(f"  {cluster:<30} {' | '.join(indices)}")
