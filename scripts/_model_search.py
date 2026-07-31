"""
Comprehensive model search: try every promising variant, report results in one table.
Goal: best TOTAL ratio (close to 1.0x) + best per-cluster WMAPE.

Train: through Dec 2025 | Test: Jan-Jun 2026

Variants tested:
  A. Original (lag-only, no SMLY)
  B. With raw SMLY
  C. SMLY dampened 70/30
  D. SMLY dampened 50/50
  E. Replace lags with SMLY + seasonal index
  F. Seasonal index only (no SMLY, no raw lags — just roll6 + seasonal_idx)
  G. Original + seasonal_index (computed from historical same-month avg)
"""
import os, sys, numpy as np, pandas as pd, warnings, time
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


def build_full_panel(df, jolts, unrate, spread_m, sent):
    """Build panel with ALL possible features. Each variant picks a subset."""
    ts = df.groupby(['region_role', 'month']).agg(
        openings=('openings', 'sum'),
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
    panel['is_q1'] = (panel['cal_quarter'] == 1).astype(float)

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
    panel['roll12'] = panel.groupby('region_role')['openings'].transform(
        lambda x: x.shift(1).rolling(12, min_periods=6).mean())
    panel['trend_3m'] = panel.groupby('region_role')['openings'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).apply(
            lambda s: (s.iloc[-1] - s.iloc[0]) / max(s.iloc[0], 1) if len(s) >= 2 else 0, raw=False))
    panel['trend_3m'] = panel['trend_3m'].fillna(0).clip(-2, 2)
    panel['yoy_change'] = panel.groupby('region_role')['openings'].transform(
        lambda x: (x - x.shift(12)) / x.shift(12).clip(lower=1))
    panel['yoy_change'] = panel['yoy_change'].fillna(0).clip(-2, 2)

    # SMLY: same-month-last-year
    panel['smly'] = panel.groupby('region_role')['openings'].shift(12)
    # Dampened SMLY variants
    panel['smly_d70'] = 0.7 * panel['smly'] + 0.3 * panel['roll6']
    panel['smly_d50'] = 0.5 * panel['smly'] + 0.5 * panel['roll6']

    # Seasonal index: for each cluster, ratio of this calendar month's historical avg to overall avg
    cluster_month_avg = panel.groupby(['region_role', 'cal_month'])['openings'].transform('mean')
    cluster_avg = panel.groupby('region_role')['openings'].transform('mean')
    panel['seasonal_idx'] = (cluster_month_avg / cluster_avg.clip(lower=1)).clip(0.2, 3.0)

    # Level vs seasonal norm
    panel['level_vs_smly'] = (panel['roll6'] / panel['smly'].clip(lower=1)).clip(0.1, 5.0)

    # Merge macro signals
    panel = panel.merge(jolts[['month', 'jolts_openings', 'jolts_yoy']], on='month', how='left')
    panel = panel.merge(unrate[['month', 'UNRATE']], on='month', how='left')
    panel = panel.merge(spread_m[['month', 'T10Y2Y']], on='month', how='left')
    panel = panel.merge(sent[['month', 'UMCSENT']], on='month', how='left')
    for col in ['jolts_openings', 'jolts_yoy', 'UNRATE', 'T10Y2Y', 'UMCSENT']:
        panel[col] = panel.groupby('region_role')[col].transform(lambda x: x.ffill())

    le_cluster = LabelEncoder()
    panel['cluster_idx'] = le_cluster.fit_transform(panel['region_role'])
    n_clusters = panel['cluster_idx'].nunique()

    return panel, le_cluster, n_clusters


def run_variant(panel_full, df_full, variant_name, feat_cols, n_clusters, test_months):
    """Run one model variant and return results."""
    # Drop rows with NaN in required features
    panel = panel_full.dropna(subset=feat_cols).copy()

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

    nc = panel['cluster_idx'].nunique()

    t0 = time.time()
    nuts = NUTS(cluster_model, target_accept_prob=0.9, max_tree_depth=10)
    mcmc = MCMC(nuts, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES,
                num_chains=NUM_CHAINS, chain_method='sequential', progress_bar=False)
    mcmc.run(jax.random.PRNGKey(42),
             cluster_idx=jnp.array(cluster_w),
             X=jnp.array(X_w), y=jnp.array(y_w),
             n_clusters=nc)
    samples = mcmc.get_samples()
    elapsed = time.time() - t0

    # Build prediction features
    cluster_last = panel.groupby(['region_role', 'cluster_idx']).last().reset_index()

    # Build SMLY lookup for prediction months
    ts_train = df_full.groupby(['region_role', 'month'])['openings'].sum().reset_index()
    smly_lookup = {}
    for _, row in ts_train.iterrows():
        rr = row['region_role']
        m = row['month']
        if m.year == 2025 and m.month <= 6:
            smly_lookup[(rr, m.month)] = row['openings']

    # Seasonal index lookup (from training data)
    ts_train['cal_month'] = ts_train['month'].dt.month
    cluster_month_avg = ts_train.groupby(['region_role', 'cal_month'])['openings'].mean()
    cluster_overall_avg = ts_train.groupby('region_role')['openings'].mean()

    forecasts = []
    for month in test_months:
        cal_m = month.month
        cal_q = ((cal_m - 1) // 3) + 1
        cl = cluster_last.copy()
        cl['month_sin'] = np.sin(2 * np.pi * cal_m / 12)
        cl['month_cos'] = np.cos(2 * np.pi * cal_m / 12)
        cl['is_q4'] = float(cal_q == 4)
        cl['is_nov_dec'] = float(cal_m in [11, 12])
        cl['is_q1'] = float(cal_q == 1)

        # SMLY features for prediction
        def get_smly(rr):
            return smly_lookup.get((rr, cal_m), cl.loc[cl['region_role'] == rr, 'roll6'].values[0]
                                   if len(cl.loc[cl['region_role'] == rr]) > 0 else 0)

        if 'smly' in feat_cols or 'smly_d70' in feat_cols or 'smly_d50' in feat_cols:
            cl['smly'] = cl['region_role'].apply(get_smly)
            cl['smly_d70'] = 0.7 * cl['smly'] + 0.3 * cl['roll6']
            cl['smly_d50'] = 0.5 * cl['smly'] + 0.5 * cl['roll6']
            cl['level_vs_smly'] = (cl['roll6'] / cl['smly'].clip(lower=1)).clip(0.1, 5.0)

        if 'seasonal_idx' in feat_cols:
            def get_sidx(rr):
                try:
                    cm_avg = cluster_month_avg.loc[(rr, cal_m)]
                    oa = cluster_overall_avg.loc[rr]
                    return min(max(cm_avg / max(oa, 1), 0.2), 3.0)
                except KeyError:
                    return 1.0
            cl['seasonal_idx'] = cl['region_role'].apply(get_sidx)

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
    actuals = df_full[df_full['month'].isin(test_months)].groupby(
        ['region_role', 'month'])['openings'].sum().reset_index()
    actuals['month_str'] = actuals['month'].astype(str)

    merged = pred_df.merge(actuals, left_on=['region_role', 'month'],
                           right_on=['region_role', 'month_str'], how='inner')
    merged['abs_error'] = np.abs(merged['predicted'] - merged['openings'])

    total_pred = merged['predicted'].sum()
    total_actual = merged['openings'].sum()
    ratio = total_pred / total_actual
    wmape = merged['abs_error'].sum() / merged['openings'].sum() * 100
    accuracy = 100 - wmape

    # US-only metrics
    us = merged[merged['region_role'].str.startswith('US')]
    us_ratio = us['predicted'].sum() / us['openings'].sum() if us['openings'].sum() > 0 else 0
    us_wmape = us['abs_error'].sum() / us['openings'].sum() * 100 if us['openings'].sum() > 0 else 0

    # IN-only metrics
    india = merged[merged['region_role'].str.startswith('IN')]
    in_ratio = india['predicted'].sum() / india['openings'].sum() if india['openings'].sum() > 0 else 0
    in_wmape = india['abs_error'].sum() / india['openings'].sum() * 100 if india['openings'].sum() > 0 else 0

    return {
        'variant': variant_name,
        'features': len(feat_cols),
        'rows': len(panel),
        'ratio': ratio,
        'wmape': wmape,
        'accuracy': accuracy,
        'us_ratio': us_ratio,
        'us_wmape': us_wmape,
        'in_ratio': in_ratio,
        'in_wmape': in_wmape,
        'time': elapsed,
    }


def main():
    print("=" * 80)
    print("  COMPREHENSIVE MODEL SEARCH")
    print("  Train: through Dec 2025 | Test: Jan-Jun 2026 (6-month blind holdout)")
    print("=" * 80)

    df, title_map = load_data()
    jolts, unrate, spread_m, sent = load_macro_signals()

    cutoff = pd.Period('2025-12', freq='M')
    df_train = df[df['month'] <= cutoff].copy()
    df_full = df.copy()  # includes 2026 actuals for comparison

    panel_full, le_cluster, n_clusters = build_full_panel(df_train, jolts, unrate, spread_m, sent)
    test_months = [pd.Period(f'2026-{m:02d}', freq='M') for m in range(1, 7)]

    macro_cols = ['jolts_openings', 'jolts_yoy', 'UNRATE', 'T10Y2Y', 'UMCSENT']
    calendar_cols = ['month_sin', 'month_cos', 'is_q4', 'is_nov_dec']

    variants = {
        'A: Original': ['lag1', 'lag2', 'lag3', 'roll3', 'roll6', 'trend_3m', 'yoy_change'] + calendar_cols + macro_cols,

        'B: +SMLY raw': ['lag1', 'lag2', 'lag3', 'roll3', 'roll6', 'trend_3m', 'yoy_change',
                         'smly', 'level_vs_smly'] + calendar_cols + macro_cols,

        'C: +SMLY damp70': ['lag1', 'lag2', 'lag3', 'roll3', 'roll6', 'trend_3m', 'yoy_change',
                            'smly_d70'] + calendar_cols + macro_cols,

        'D: +SMLY damp50': ['lag1', 'lag2', 'lag3', 'roll3', 'roll6', 'trend_3m', 'yoy_change',
                            'smly_d50'] + calendar_cols + macro_cols,

        'E: +SeasonalIdx': ['lag1', 'lag2', 'lag3', 'roll3', 'roll6', 'trend_3m', 'yoy_change',
                            'seasonal_idx'] + calendar_cols + macro_cols,

        'F: SMLY+SIdx': ['lag1', 'roll6', 'trend_3m', 'yoy_change',
                         'smly_d70', 'seasonal_idx'] + calendar_cols + macro_cols,

        'G: Roll12+SIdx': ['roll6', 'roll12', 'trend_3m', 'yoy_change',
                           'seasonal_idx', 'is_q1'] + calendar_cols + macro_cols,

        'H: Kitchen sink': ['lag1', 'lag2', 'lag3', 'roll3', 'roll6', 'roll12', 'trend_3m', 'yoy_change',
                            'smly_d70', 'seasonal_idx', 'is_q1'] + calendar_cols + macro_cols,
    }

    results = []
    for name, feat_cols in variants.items():
        print(f"\n  Running {name} ({len(feat_cols)} features)...")
        try:
            r = run_variant(panel_full, df_full, name, feat_cols, n_clusters, test_months)
            results.append(r)
            print(f"    ratio={r['ratio']:.2f}x  WMAPE={r['wmape']:.1f}%  US={r['us_ratio']:.2f}x/{r['us_wmape']:.0f}%  IN={r['in_ratio']:.2f}x/{r['in_wmape']:.0f}%  ({r['time']:.0f}s)")
        except Exception as e:
            print(f"    FAILED: {e}")

    # Results table
    print("\n\n" + "=" * 100)
    print("  RESULTS — sorted by combined score (ratio closeness + accuracy)")
    print("=" * 100)
    print(f"  {'Variant':<22} {'Ratio':>6} {'WMAPE':>7} {'Acc':>6} {'US ratio':>9} {'US WMAPE':>9} {'IN ratio':>9} {'IN WMAPE':>9} {'Score':>6}")
    print(f"  {'-'*88}")

    for r in results:
        # Score: penalize both under and over-prediction, reward low WMAPE
        # ratio_penalty: distance from 1.0 (0 = perfect, higher = worse)
        ratio_penalty = abs(1.0 - r['ratio']) * 100
        # Combined score: lower is better
        r['score'] = r['wmape'] * 0.6 + ratio_penalty * 0.4

    results.sort(key=lambda x: x['score'])

    for r in results:
        marker = ' ★' if r == results[0] else ''
        print(f"  {r['variant']:<22} {r['ratio']:>5.2f}x {r['wmape']:>6.1f}% {r['accuracy']:>5.1f}% {r['us_ratio']:>8.2f}x {r['us_wmape']:>8.1f}% {r['in_ratio']:>8.2f}x {r['in_wmape']:>8.1f}% {r['score']:>5.1f}{marker}")

    best = results[0]
    print(f"\n  ★ BEST: {best['variant']}")
    print(f"    Total: {best['ratio']:.2f}x ratio, {best['accuracy']:.1f}% accuracy")
    print(f"    US:    {best['us_ratio']:.2f}x ratio, {100-best['us_wmape']:.1f}% accuracy")
    print(f"    IN:    {best['in_ratio']:.2f}x ratio, {100-best['in_wmape']:.1f}% accuracy")


if __name__ == '__main__':
    main()
