"""
Holdout test with SMLY (same-month-last-year) as an additional anchor feature.
The hypothesis: SMLY gives the model a seasonality-aware baseline so it doesn't
just extrapolate the Q4 dip forward.
"""
import os, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax, jax.numpy as jnp, numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

sys.path.insert(0, os.path.dirname(__file__))
from generate_hier_forecasts import (
    load_data, load_macro_signals, cluster_model,
    NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS, CI_LO_PCT, CI_HI_PCT
)


def build_panel_with_smly(df, jolts, unrate, spread_m, sent):
    """Build panel with same-month-last-year as anchor feature."""
    print("  Building cluster-level panel (with SMLY anchor)...")

    ts = df.groupby(['region_role', 'month']).agg(
        openings=('openings', 'sum'),
        n_companies=('company_name', 'nunique'),
    ).reset_index()

    cluster_months = ts.groupby('region_role')['month'].nunique()
    viable = cluster_months[cluster_months >= 12].index.tolist()
    panel = ts[ts['region_role'].isin(viable)].copy()

    panel['cal_month'] = panel['month'].dt.month
    panel['cal_quarter'] = ((panel['cal_month'] - 1) // 3) + 1
    panel['month_sin'] = np.sin(2 * np.pi * panel['cal_month'] / 12)
    panel['month_cos'] = np.cos(2 * np.pi * panel['cal_month'] / 12)
    panel['is_q4'] = (panel['cal_quarter'] == 4).astype(float)
    panel['is_nov_dec'] = panel['cal_month'].isin([11, 12]).astype(float)

    panel['year'] = panel['month'].dt.year
    weight_map = {2026: 1.0, 2025: 1.0, 2024: 0.7, 2023: 0.5, 2022: 0.3}
    panel['sample_weight'] = panel['year'].map(lambda y: weight_map.get(y, 0.15))

    panel = panel.sort_values(['region_role', 'month']).reset_index(drop=True)
    panel['lag1'] = panel.groupby('region_role')['openings'].shift(1)
    panel['lag2'] = panel.groupby('region_role')['openings'].shift(2)
    panel['lag3'] = panel.groupby('region_role')['openings'].shift(3)
    panel['roll3'] = panel.groupby('region_role')['openings'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    panel['roll6'] = panel.groupby('region_role')['openings'].transform(
        lambda x: x.shift(1).rolling(6, min_periods=3).mean())
    panel['trend_3m'] = panel.groupby('region_role')['openings'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).apply(
            lambda s: (s.iloc[-1] - s.iloc[0]) / max(s.iloc[0], 1) if len(s) >= 2 else 0, raw=False))
    panel['trend_3m'] = panel['trend_3m'].fillna(0).clip(-2, 2)
    panel['yoy_change'] = panel.groupby('region_role')['openings'].transform(
        lambda x: (x - x.shift(12)) / x.shift(12).clip(lower=1))
    panel['yoy_change'] = panel['yoy_change'].fillna(0).clip(-2, 2)

    # ─── KEY NEW FEATURE: Same-month-last-year (SMLY) ───
    panel['smly'] = panel.groupby('region_role')['openings'].shift(12)
    # Also add ratio of roll6 to SMLY (captures whether current level is above/below seasonal norm)
    panel['level_vs_smly'] = panel['roll6'] / panel['smly'].clip(lower=1)
    panel['level_vs_smly'] = panel['level_vs_smly'].fillna(1.0).clip(0.1, 5.0)

    # Merge macro signals
    panel = panel.merge(jolts[['month', 'jolts_openings', 'jolts_yoy']], on='month', how='left')
    panel = panel.merge(unrate[['month', 'UNRATE']], on='month', how='left')
    panel = panel.merge(spread_m[['month', 'T10Y2Y']], on='month', how='left')
    panel = panel.merge(sent[['month', 'UMCSENT']], on='month', how='left')

    for col in ['jolts_openings', 'jolts_yoy', 'UNRATE', 'T10Y2Y', 'UMCSENT']:
        panel[col] = panel.groupby('region_role')[col].transform(lambda x: x.ffill())

    panel = panel.dropna(subset=['lag1', 'lag2', 'lag3', 'roll3', 'roll6', 'smly',
                                  'jolts_openings', 'UNRATE', 'T10Y2Y', 'UMCSENT']).copy()

    le_cluster = LabelEncoder()
    panel['cluster_idx'] = le_cluster.fit_transform(panel['region_role'])
    n_clusters = panel['cluster_idx'].nunique()

    print(f"  Panel: {len(panel):,} rows, {n_clusters} clusters")
    return panel, le_cluster, n_clusters


def main():
    print("=" * 70)
    print("  HOLDOUT TEST — with SMLY anchor feature")
    print("  Train: through Dec 2025 | Test: Jan-Jun 2026")
    print("=" * 70)

    df, title_map = load_data()
    jolts, unrate, spread_m, sent = load_macro_signals()

    cutoff = pd.Period('2025-12', freq='M')
    df_train = df[df['month'] <= cutoff].copy()
    print(f"\n  Train: {len(df_train):,} records through {cutoff}")

    panel, le_cluster, n_clusters = build_panel_with_smly(df_train, jolts, unrate, spread_m, sent)

    test_months = [pd.Period(f'2026-{m:02d}', freq='M') for m in range(1, 7)]
    actuals = df[df['month'].isin(test_months)].groupby(['region_role', 'month'])['openings'].sum().reset_index()
    print(f"  Actuals: {len(actuals)} cluster-months, {actuals['openings'].sum():,} total")

    # Features — now includes smly and level_vs_smly
    feat_cols = ['lag1', 'lag2', 'lag3', 'roll3', 'roll6', 'trend_3m', 'yoy_change',
                 'smly', 'level_vs_smly',
                 'month_sin', 'month_cos', 'is_q4', 'is_nov_dec',
                 'jolts_openings', 'jolts_yoy', 'UNRATE', 'T10Y2Y', 'UMCSENT']

    X_train = panel[feat_cols].values.astype(np.float32)
    y_train = panel['openings'].values.astype(np.float32)
    weights = panel['sample_weight'].values.astype(np.float32)
    cluster_train = panel['cluster_idx'].values.astype(np.int32)

    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-6
    X_train_s = (X_train - X_mean) / X_std
    y_train_log = np.log1p(y_train).astype(np.float32)

    repeat_map = {1.0: 2, 0.7: 2, 0.5: 1, 0.3: 1, 0.15: 1}
    repeats = np.array([repeat_map.get(w, 1) for w in weights])
    X_w = np.repeat(X_train_s, repeats, axis=0)
    y_w = np.repeat(y_train_log, repeats)
    cluster_w = np.repeat(cluster_train, repeats)

    print(f"\n  Features: {len(feat_cols)} (added: smly, level_vs_smly)")
    print(f"  Fitting model...")
    nuts = NUTS(cluster_model, target_accept_prob=0.9, max_tree_depth=10)
    mcmc = MCMC(nuts, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES,
                num_chains=NUM_CHAINS, chain_method='sequential', progress_bar=True)
    mcmc.run(jax.random.PRNGKey(42),
             cluster_idx=jnp.array(cluster_w),
             X=jnp.array(X_w), y=jnp.array(y_w),
             n_clusters=n_clusters)
    samples = mcmc.get_samples()

    # Check which features the model weights most
    beta_median = np.median(np.array(samples['beta']), axis=0)
    print("\n  Feature weights (standardized beta):")
    for i, col in enumerate(feat_cols):
        print(f"    {col:<20} {beta_median[i]:+.3f}")

    # Predict Jan-Jun 2026
    print("\n  Generating predictions...")
    cluster_last = panel.groupby(['region_role', 'cluster_idx']).agg(
        lag1=('lag1', 'last'), lag2=('lag2', 'last'), lag3=('lag3', 'last'),
        roll3=('roll3', 'last'), roll6=('roll6', 'last'),
        trend_3m=('trend_3m', 'last'), yoy_change=('yoy_change', 'last'),
        jolts_openings=('jolts_openings', 'last'), jolts_yoy=('jolts_yoy', 'last'),
        UNRATE=('UNRATE', 'last'), T10Y2Y=('T10Y2Y', 'last'), UMCSENT=('UMCSENT', 'last'),
    ).reset_index()

    # For SMLY at prediction time: need same-month-last-year values
    # Build a lookup: for each cluster, what was Jan/Feb/.../Jun 2025?
    smly_lookup = {}
    ts_full = df_train.groupby(['region_role', 'month'])['openings'].sum().reset_index()
    for _, row in ts_full.iterrows():
        rr = row['region_role']
        m = row['month']
        if m.year == 2025 and m.month <= 6:
            smly_lookup[(rr, m.month)] = row['openings']

    forecasts = []
    for month in test_months:
        cal_m = month.month
        cal_q = ((cal_m - 1) // 3) + 1
        cl = cluster_last.copy()
        cl['month_sin'] = np.sin(2 * np.pi * cal_m / 12)
        cl['month_cos'] = np.cos(2 * np.pi * cal_m / 12)
        cl['is_q4'] = float(cal_q == 4)
        cl['is_nov_dec'] = float(cal_m in [11, 12])
        # SMLY: same month from 2025
        cl['smly'] = cl['region_role'].apply(lambda rr: smly_lookup.get((rr, cal_m), cl.loc[cl['region_role'] == rr, 'roll6'].values[0] if len(cl.loc[cl['region_role'] == rr]) > 0 else 0))
        cl['level_vs_smly'] = (cl['roll6'] / cl['smly'].clip(lower=1)).clip(0.1, 5.0)

        X_pred = cl[feat_cols].values.astype(np.float32)
        X_pred_s = (X_pred - X_mean) / X_std
        cluster_pred = cl['cluster_idx'].values.astype(np.int32)

        pred_log = (samples['mu_global'][:, None] +
                    samples['alpha_cluster'][:, cluster_pred] +
                    jnp.dot(samples['beta'], jnp.array(X_pred_s).T))
        pred_arr = np.maximum(0, np.expm1(np.clip(np.array(pred_log), None, 10)))
        point = np.median(pred_arr, axis=0)

        for i in range(len(cl)):
            forecasts.append({
                'region_role': cl.iloc[i]['region_role'],
                'month': str(month),
                'predicted': max(0, round(float(point[i]))),
            })

    pred_df = pd.DataFrame(forecasts)
    actuals['month_str'] = actuals['month'].astype(str)

    merged = pred_df.merge(actuals, left_on=['region_role', 'month'],
                           right_on=['region_role', 'month_str'], how='inner')
    merged['abs_error'] = np.abs(merged['predicted'] - merged['openings'])

    total_pred = merged['predicted'].sum()
    total_actual = merged['openings'].sum()
    wmape = merged['abs_error'].sum() / merged['openings'].sum() * 100
    ratio = total_pred / total_actual

    print(f"\n  {'=' * 60}")
    print(f"  HOLDOUT RESULTS — with SMLY anchor")
    print(f"  {'=' * 60}")
    print(f"  Total predicted:         {total_pred:,.0f}")
    print(f"  Total actual:            {total_actual:,.0f}")
    print(f"  Ratio (pred/actual):     {ratio:.2f}x")
    print(f"  WMAPE:                   {wmape:.1f}%")
    print(f"  Accuracy (1-WMAPE):      {100 - wmape:.1f}%")
    print(f"  {'=' * 60}")

    # Per-month
    print(f"\n  Monthly:")
    for m_str in [str(m) for m in test_months]:
        sub = merged[merged['month_x'] == m_str] if 'month_x' in merged.columns else merged[merged['month'] == m_str]
        if len(sub) == 0: continue
        mp = sub['predicted'].sum()
        ma = sub['openings'].sum()
        mw = sub['abs_error'].sum() / sub['openings'].sum() * 100
        print(f"    {m_str}: pred={mp:>5,}  actual={ma:>5,}  ratio={mp / ma:.2f}x  WMAPE={mw:.1f}%")

    # Per-region
    print(f"\n  By region:")
    for region in ['US', 'IN', 'LATAM & Others']:
        sub = merged[merged['region_role'].str.startswith(region)]
        if len(sub) == 0: continue
        rp = sub['predicted'].sum()
        ra = sub['openings'].sum()
        rw = sub['abs_error'].sum() / sub['openings'].sum() * 100
        print(f"    {region:<20} pred={rp:>5,}  actual={ra:>5,}  ratio={rp / ra:.2f}x  WMAPE={rw:.1f}%")


if __name__ == '__main__':
    main()
