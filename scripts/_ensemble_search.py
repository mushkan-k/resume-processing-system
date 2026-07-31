"""
Ensemble test: Blend Original (A) and Roll12+SIdx (G) predictions.
Try multiple blend ratios to find the sweet spot.
"""
import os, sys, numpy as np, pandas as pd, warnings, time
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
    load_data, load_macro_signals, cluster_model,
    NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS
)
from _model_search import build_full_panel


def fit_and_predict_variant(panel_full, df_full, feat_cols, test_months):
    """Fit one variant and return per-cluster-month predictions."""
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

    nuts = NUTS(cluster_model, target_accept_prob=0.9, max_tree_depth=10)
    mcmc = MCMC(nuts, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES,
                num_chains=NUM_CHAINS, chain_method='sequential', progress_bar=False)
    mcmc.run(jax.random.PRNGKey(42),
             cluster_idx=jnp.array(cluster_w),
             X=jnp.array(X_w), y=jnp.array(y_w),
             n_clusters=nc)
    samples = mcmc.get_samples()

    cluster_last = panel.groupby(['region_role', 'cluster_idx']).last().reset_index()

    # SMLY lookup
    ts_train = df_full[df_full['month'] <= pd.Period('2025-12', freq='M')].groupby(
        ['region_role', 'month'])['openings'].sum().reset_index()
    smly_lookup = {}
    for _, row in ts_train.iterrows():
        if row['month'].year == 2025 and row['month'].month <= 6:
            smly_lookup[(row['region_role'], row['month'].month)] = row['openings']

    # Seasonal index lookup
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

        if 'smly' in feat_cols or 'smly_d70' in feat_cols:
            cl['smly'] = cl['region_role'].apply(
                lambda rr: smly_lookup.get((rr, cal_m), cl.loc[cl['region_role'] == rr, 'roll6'].values[0]
                           if len(cl.loc[cl['region_role'] == rr]) > 0 else 0))
            cl['smly_d70'] = 0.7 * cl['smly'] + 0.3 * cl['roll6']
            cl['level_vs_smly'] = (cl['roll6'] / cl['smly'].clip(lower=1)).clip(0.1, 5.0)

        if 'seasonal_idx' in feat_cols:
            def get_sidx(rr):
                try:
                    return min(max(cluster_month_avg.loc[(rr, cal_m)] / max(cluster_overall_avg.loc[rr], 1), 0.2), 3.0)
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
                'predicted': max(0, float(point[i])),
            })

    return pd.DataFrame(forecasts)


def evaluate(pred_df, df_full, test_months, label):
    actuals = df_full[df_full['month'].isin(test_months)].groupby(
        ['region_role', 'month'])['openings'].sum().reset_index()
    actuals['month_str'] = actuals['month'].astype(str)

    merged = pred_df.merge(actuals, left_on=['region_role', 'month'],
                           right_on=['region_role', 'month_str'], how='inner')
    merged['abs_error'] = np.abs(merged['predicted'] - merged['openings'])

    total_pred = merged['predicted'].sum()
    total_actual = merged['openings'].sum()
    ratio = total_pred / total_actual
    wmape = merged['abs_error'].sum() / total_actual * 100

    us = merged[merged['region_role'].str.startswith('US')]
    us_ratio = us['predicted'].sum() / us['openings'].sum()
    us_wmape = us['abs_error'].sum() / us['openings'].sum() * 100

    india = merged[merged['region_role'].str.startswith('IN')]
    in_ratio = india['predicted'].sum() / india['openings'].sum()
    in_wmape = india['abs_error'].sum() / india['openings'].sum() * 100

    return {
        'label': label, 'ratio': ratio, 'wmape': wmape,
        'us_ratio': us_ratio, 'us_wmape': us_wmape,
        'in_ratio': in_ratio, 'in_wmape': in_wmape,
    }


def main():
    print("=" * 80)
    print("  ENSEMBLE MODEL SEARCH")
    print("  Blending Original (A) + Roll12+SIdx (G)")
    print("  Train: through Dec 2025 | Test: Jan-Jun 2026")
    print("=" * 80)

    df, title_map = load_data()
    jolts, unrate, spread_m, sent = load_macro_signals()

    cutoff = pd.Period('2025-12', freq='M')
    df_train = df[df['month'] <= cutoff].copy()
    df_full = df.copy()

    panel_full, le_cluster, n_clusters = build_full_panel(df_train, jolts, unrate, spread_m, sent)
    test_months = [pd.Period(f'2026-{m:02d}', freq='M') for m in range(1, 7)]

    macro_cols = ['jolts_openings', 'jolts_yoy', 'UNRATE', 'T10Y2Y', 'UMCSENT']
    calendar_cols = ['month_sin', 'month_cos', 'is_q4', 'is_nov_dec']

    feat_A = ['lag1', 'lag2', 'lag3', 'roll3', 'roll6', 'trend_3m', 'yoy_change'] + calendar_cols + macro_cols
    feat_G = ['roll6', 'roll12', 'trend_3m', 'yoy_change', 'seasonal_idx', 'is_q1'] + calendar_cols + macro_cols

    print("\n  Fitting Model A (Original)...")
    pred_A = fit_and_predict_variant(panel_full, df_full, feat_A, test_months)
    print("  Fitting Model G (Roll12+SIdx)...")
    pred_G = fit_and_predict_variant(panel_full, df_full, feat_G, test_months)

    # Merge predictions from both models
    merged_preds = pred_A.merge(pred_G, on=['region_role', 'month'], suffixes=('_A', '_G'))

    results = []

    # Pure models
    results.append(evaluate(pred_A, df_full, test_months, 'A only (Original)'))
    results.append(evaluate(pred_G, df_full, test_months, 'G only (Roll12+SIdx)'))

    # Blends
    for w_a in [0.7, 0.6, 0.5, 0.4, 0.3]:
        w_g = 1.0 - w_a
        blended = merged_preds.copy()
        blended['predicted'] = w_a * blended['predicted_A'] + w_g * blended['predicted_G']
        blended['predicted'] = blended['predicted'].round()
        label = f'Blend A={int(w_a*100)}/G={int(w_g*100)}'
        results.append(evaluate(blended[['region_role', 'month', 'predicted']], df_full, test_months, label))

    # Max of both (always pick the higher prediction — anti-under-prediction)
    max_pred = merged_preds.copy()
    max_pred['predicted'] = np.maximum(max_pred['predicted_A'], max_pred['predicted_G']).round()
    results.append(evaluate(max_pred[['region_role', 'month', 'predicted']], df_full, test_months, 'Max(A,G)'))

    # Weighted by region: A for US, G for IN
    region_blend = merged_preds.copy()
    is_us = region_blend['region_role'].str.startswith('US')
    is_in = region_blend['region_role'].str.startswith('IN')
    is_other = ~is_us & ~is_in
    # Use more G for US (to fix under-prediction), more A for IN (to reduce noise)
    region_blend.loc[is_us, 'predicted'] = (0.4 * region_blend.loc[is_us, 'predicted_A'] + 0.6 * region_blend.loc[is_us, 'predicted_G']).round()
    region_blend.loc[is_in, 'predicted'] = (0.6 * region_blend.loc[is_in, 'predicted_A'] + 0.4 * region_blend.loc[is_in, 'predicted_G']).round()
    region_blend.loc[is_other, 'predicted'] = (0.5 * region_blend.loc[is_other, 'predicted_A'] + 0.5 * region_blend.loc[is_other, 'predicted_G']).round()
    results.append(evaluate(region_blend[['region_role', 'month', 'predicted']], df_full, test_months, 'Region-tuned'))

    # Region-tuned v2: A=50/G=50 for US, A=70/G=30 for IN
    rb2 = merged_preds.copy()
    rb2.loc[is_us, 'predicted'] = (0.5 * rb2.loc[is_us, 'predicted_A'] + 0.5 * rb2.loc[is_us, 'predicted_G']).round()
    rb2.loc[is_in, 'predicted'] = (0.7 * rb2.loc[is_in, 'predicted_A'] + 0.3 * rb2.loc[is_in, 'predicted_G']).round()
    rb2.loc[is_other, 'predicted'] = (0.5 * rb2.loc[is_other, 'predicted_A'] + 0.5 * rb2.loc[is_other, 'predicted_G']).round()
    results.append(evaluate(rb2[['region_role', 'month', 'predicted']], df_full, test_months, 'Region-tuned v2'))

    # Print results
    print(f"\n  {'=' * 95}")
    print(f"  {'Variant':<22} {'Ratio':>7} {'WMAPE':>7} {'Acc':>6}  {'US rat':>7} {'US WM':>7} {'US Acc':>7}  {'IN rat':>7} {'IN WM':>7}")
    print(f"  {'-' * 93}")

    # Score and sort
    for r in results:
        ratio_penalty = abs(1.0 - r['ratio']) * 100
        us_ratio_pen = abs(1.0 - r['us_ratio']) * 100
        # Weighted score: overall ratio closeness (30%) + overall WMAPE (30%) + US ratio (20%) + US WMAPE (20%)
        r['score'] = ratio_penalty * 0.3 + r['wmape'] * 0.3 + us_ratio_pen * 0.2 + r['us_wmape'] * 0.2
    results.sort(key=lambda x: x['score'])

    for r in results:
        marker = ' ★' if r == results[0] else ''
        print(f"  {r['label']:<22} {r['ratio']:>6.2f}x {r['wmape']:>6.1f}% {100-r['wmape']:>5.1f}%  {r['us_ratio']:>6.2f}x {r['us_wmape']:>6.1f}% {100-r['us_wmape']:>6.1f}%  {r['in_ratio']:>6.2f}x {r['in_wmape']:>6.1f}%{marker}")

    best = results[0]
    print(f"\n  ★ BEST: {best['label']}")
    print(f"    Overall: ratio={best['ratio']:.2f}x, accuracy={100-best['wmape']:.1f}%")
    print(f"    US:      ratio={best['us_ratio']:.2f}x, accuracy={100-best['us_wmape']:.1f}%")
    print(f"    IN:      ratio={best['in_ratio']:.2f}x, accuracy={100-best['in_wmape']:.1f}%")


if __name__ == '__main__':
    main()
