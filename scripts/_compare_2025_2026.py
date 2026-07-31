import pandas as pd, numpy as np, os

df = pd.read_pickle(os.path.join('data','clean_42k_v1.pkl'))
ttc = pd.read_pickle(os.path.join('data','title_to_cluster.pkl'))
df['role_cluster'] = df['title'].map(dict(zip(ttc['raw_title'], ttc['role_cluster'])))
df = df.dropna(subset=['role_cluster'])
df['issue_date'] = pd.to_datetime(df['issue_date'])
df['month'] = df['issue_date'].dt.to_period('M')
df['year'] = df['issue_date'].dt.year
df['cal_month'] = df['issue_date'].dt.month

monthly = df[df['year'].isin([2025, 2026])].groupby(['year', 'cal_month'])['openings'].sum().reset_index()
monthly = monthly.pivot(index='cal_month', columns='year', values='openings').fillna(0)

print("\n  Month-by-Month Comparison: 2025 Actual vs 2026 (Actual + Predicted)")
print("  " + "=" * 65)
print(f"  {'Month':>7}  {'2025 Actual':>12}  {'2026 Actual':>12}  {'2026 Pred':>10}  {'YoY':>6}")
print("  " + "-" * 65)

pred_h2 = {7: 659, 8: 681, 9: 703, 10: 647, 11: 538, 12: 521}

for m in range(1, 13):
    v25 = int(monthly.loc[m, 2025]) if m in monthly.index and 2025 in monthly.columns else 0
    v26 = int(monthly.loc[m, 2026]) if m in monthly.index and 2026 in monthly.columns else 0
    pred = pred_h2.get(m, None)

    if m <= 6:
        yoy = f"{v26/v25:.0%}" if v25 > 0 else "N/A"
        print(f"  {m:>7}  {v25:>12,}  {v26:>12,}  {'':>10}  {yoy:>6}")
    else:
        yoy25 = v25
        yoy_str = f"{pred/v25:.0%}" if v25 > 0 and pred else "N/A"
        label = f"{pred:,}" if pred else ""
        print(f"  {m:>7}  {v25:>12,}  {'':>12}  {label:>10}  {yoy_str:>6}")

print("  " + "-" * 65)

h1_25 = monthly.loc[1:6, 2025].sum() if 2025 in monthly.columns else 0
h2_25 = monthly.loc[7:12, 2025].sum() if 2025 in monthly.columns else 0
h1_26 = monthly.loc[1:6, 2026].sum() if 2026 in monthly.columns else 0
h2_26_pred = sum(pred_h2.values())

print(f"\n  2025:  H1={int(h1_25):>6,}   H2={int(h2_25):>6,}   Full={int(h1_25+h2_25):>6,}   H2/H1={h2_25/h1_25:.2f}x")
print(f"  2026:  H1={int(h1_26):>6,}   H2={int(h2_26_pred):>6,}*  Full={int(h1_26+h2_26_pred):>6,}   H2/H1={h2_26_pred/h1_26:.2f}x")
print(f"  YoY:   H1={h1_26/h1_25:.0%}      H2={h2_26_pred/h2_25:.0%}*     Full={int(h1_26+h2_26_pred)/int(h1_25+h2_25):.0%}")
print(f"\n  * H2 2026 = model predicted")
print(f"  2025 had a natural H2/H1 ratio of {h2_25/h1_25:.2f}x")
print(f"  2026 model predicts H2/H1 of {h2_26_pred/h1_26:.2f}x — {'consistent' if abs(h2_26_pred/h1_26 - h2_25/h1_25) < 0.15 else 'divergent'} with 2025 pattern")
