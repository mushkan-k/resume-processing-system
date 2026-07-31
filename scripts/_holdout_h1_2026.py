"""
Holdout validation: Train through Dec 2025, predict Jan-Jun 2026, compare to actuals.
This is the strongest test — full 6-month forward prediction vs real data.
"""
import os, sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax, jax.numpy as jnp, numpyro
from numpyro.infer import MCMC, NUTS
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

sys.path.insert(0, os.path.dirname(__file__))
from generate_hier_forecasts import (
    load_data, load_macro_signals, build_cluster_panel, cluster_model,
    NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS, CI_LO_PCT, CI_HI_PCT
)


def main():
    print("=" * 70)
    print("  HOLDOUT VALIDATION — Train through Dec 2025, Predict H1 2026")
    print("=" * 70)

    df, title_map = load_data()
    jolts, unrate, spread_m, sent = load_macro_signals()

    # Split: train through Dec 2025
    cutoff = pd.Period('2025-12', freq='M')
    df_train = df[df['month'] <= cutoff].copy()
    print(f"\n  Train: {len(df_train):,} records through {cutoff}")
    print(f"  Test:  Jan-Jun 2026 actuals")

    # Build panel from TRAIN only
    panel, le_cluster, n_clusters = build_cluster_panel(df_train, jolts, unrate, spread_m, sent)

    # Build actuals for Jan-Jun 2026
    test_months = [pd.Period(f'2026-{m:02d}', freq='M') for m in range(1, 7)]
    actuals = df[df['month'].isin(test_months)].groupby(['region_role', 'month'])['openings'].sum().reset_index()
    actual_total = actuals['openings'].sum()
    print(f"  Actuals: {len(actuals)} cluster-months, {actual_total:,} total openings")

    # Fit model
    feat_cols = ['lag1', 'lag2', 'lag3', 'roll3', 'roll6', 'trend_3m', 'yoy_change',
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

    print(f"\n  Fitting model ({NUM_CHAINS} chains x {NUM_SAMPLES} samples)...")
    nuts = NUTS(cluster_model, target_accept_prob=0.9, max_tree_depth=10)
    mcmc = MCMC(nuts, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES,
                num_chains=NUM_CHAINS, chain_method='sequential', progress_bar=True)
    mcmc.run(jax.random.PRNGKey(42),
             cluster_idx=jnp.array(cluster_w),
             X=jnp.array(X_w), y=jnp.array(y_w),
             n_clusters=n_clusters)
    samples = mcmc.get_samples()

    # Predict Jan-Jun 2026
    print("  Generating H1 2026 predictions...")
    cluster_last = panel.groupby(['region_role', 'cluster_idx']).agg(
        lag1=('lag1', 'last'), lag2=('lag2', 'last'), lag3=('lag3', 'last'),
        roll3=('roll3', 'last'), roll6=('roll6', 'last'),
        trend_3m=('trend_3m', 'last'), yoy_change=('yoy_change', 'last'),
        jolts_openings=('jolts_openings', 'last'), jolts_yoy=('jolts_yoy', 'last'),
        UNRATE=('UNRATE', 'last'), T10Y2Y=('T10Y2Y', 'last'), UMCSENT=('UMCSENT', 'last'),
    ).reset_index()

    forecasts = []
    for month in test_months:
        cal_m = month.month
        cal_q = ((cal_m - 1) // 3) + 1
        cl = cluster_last.copy()
        cl['month_sin'] = np.sin(2 * np.pi * cal_m / 12)
        cl['month_cos'] = np.cos(2 * np.pi * cal_m / 12)
        cl['is_q4'] = float(cal_q == 4)
        cl['is_nov_dec'] = float(cal_m in [11, 12])

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

    # Merge and compute accuracy
    merged = pred_df.merge(actuals, left_on=['region_role', 'month'],
                           right_on=['region_role', 'month_str'], how='inner')
    merged['abs_error'] = np.abs(merged['predicted'] - merged['openings'])
    merged['ape'] = np.where(merged['openings'] > 0,
                             merged['abs_error'] / merged['openings'], np.nan)

    total_pred = merged['predicted'].sum()
    total_actual = merged['openings'].sum()
    wmape = merged['abs_error'].sum() / merged['openings'].sum() * 100
    ratio = total_pred / total_actual

    print(f"\n  {'=' * 60}")
    print(f"  HOLDOUT RESULTS — H1 2026 (Jan-Jun)")
    print(f"  Trained on: Jan 2022 – Dec 2025 only")
    print(f"  {'=' * 60}")
    print(f"  Matched cluster-months:  {len(merged)}")
    print(f"  Total predicted:         {total_pred:,.0f}")
    print(f"  Total actual:            {total_actual:,.0f}")
    print(f"  Ratio (pred/actual):     {ratio:.2f}x")
    print(f"  WMAPE:                   {wmape:.1f}%")
    print(f"  Accuracy (1-WMAPE):      {100-wmape:.1f}%")
    print(f"  Median APE:              {np.nanmedian(merged['ape'])*100:.1f}%")
    print(f"  {'=' * 60}")

    # Per-month breakdown
    print(f"\n  Monthly breakdown:")
    print(f"  {'Month':<10} {'Predicted':>10} {'Actual':>10} {'Ratio':>8} {'WMAPE':>8}")
    print(f"  {'-'*48}")
    for m in ['2026-01','2026-02','2026-03','2026-04','2026-05','2026-06']:
        sub = merged[merged['month_x'] == m] if 'month_x' in merged.columns else merged[merged['month'] == m]
        if len(sub) == 0:
            continue
        mp = sub['predicted'].sum()
        ma = sub['openings'].sum()
        mw = sub['abs_error'].sum() / sub['openings'].sum() * 100
        print(f"  {m:<10} {mp:>10,} {ma:>10,} {mp/ma:>7.2f}x {mw:>7.1f}%")

    # Per-region breakdown
    print(f"\n  By region:")
    for region in ['US', 'IN', 'LATAM & Others']:
        sub = merged[merged['region_role'].str.startswith(region)]
        if len(sub) == 0:
            continue
        rp = sub['predicted'].sum()
        ra = sub['openings'].sum()
        rw = sub['abs_error'].sum() / sub['openings'].sum() * 100
        print(f"  {region:<20} pred={rp:>5,}  actual={ra:>5,}  ratio={rp/ra:.2f}x  WMAPE={rw:.1f}%")

    # Top 10 biggest clusters — accuracy
    print(f"\n  Top 10 clusters by volume — accuracy:")
    cluster_agg = merged.groupby('region_role').agg(
        pred=('predicted','sum'), actual=('openings','sum'), err=('abs_error','sum')
    ).reset_index()
    cluster_agg['wmape'] = cluster_agg['err'] / cluster_agg['actual'] * 100
    cluster_agg['ratio'] = cluster_agg['pred'] / cluster_agg['actual']
    top10 = cluster_agg.nlargest(10, 'actual')
    for _, r in top10.iterrows():
        name = r['region_role']
        print(f"  {name:<40} pred={int(r['pred']):>5}  actual={int(r['actual']):>5}  ratio={r['ratio']:.2f}x  WMAPE={r['wmape']:.0f}%")


if __name__ == '__main__':
    main()
