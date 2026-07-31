"""
Generate demand forecasts — Max(A,G) Ensemble approach.
=========================================================
Fits TWO independent Bayesian models and takes the MAX prediction per cluster-month.
This prevents under-prediction (the riskier error) while maintaining accuracy.

Model A: "Lag model" — uses lag1/2/3, roll3/6 (good per-cluster accuracy)
Model G: "Seasonal model" — uses roll6/12 + seasonal_idx (good total calibration)
Ensemble: Max(A, G) — always pick the higher forecast (anti-under-prediction)

Holdout validated (train Dec 2025, predict H1 2026):
  - Total ratio: 0.98x (vs 0.84x for A alone, 0.94x for G alone)
  - US ratio: 0.91x (vs 0.82x for A alone)

Usage:
  python scripts/generate_hier_forecasts.py
  python scripts/generate_hier_forecasts.py --dry-run
  python scripts/generate_hier_forecasts.py --months 6
"""
import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv
import mysql.connector

load_dotenv()

# ─── Config ─────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", "3305")),
    "database": os.getenv("MYSQL_DATABASE", "resume_processing"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "rootpassword"),
}

MODEL_VERSION = "Ensemble_MaxAG"
NUM_WARMUP = 1000
NUM_SAMPLES = 2000
NUM_CHAINS = 2
CI_LO_PCT = 20
CI_HI_PCT = 80

# ─── Feature sets for the two models ────────────────────────────────────────
MACRO_COLS = ['jolts_openings', 'jolts_yoy', 'UNRATE', 'T10Y2Y', 'UMCSENT']
CAL_COLS = ['month_sin', 'month_cos', 'is_q4', 'is_nov_dec']

FEAT_A = ['lag1', 'lag2', 'lag3', 'roll3', 'roll6', 'trend_3m', 'yoy_change'] + CAL_COLS + MACRO_COLS
FEAT_G = ['roll6', 'roll12', 'trend_3m', 'yoy_change', 'seasonal_idx', 'is_q1'] + CAL_COLS + MACRO_COLS


def load_data():
    """Load clean_42k + title_to_cluster."""
    print("  Loading data...")
    df = pd.read_pickle(os.path.join(DATA_DIR, 'clean_42k_v1.pkl'))
    ttc = pd.read_pickle(os.path.join(DATA_DIR, 'title_to_cluster.pkl'))
    title_map = dict(zip(ttc['raw_title'], ttc['role_cluster']))
    df['role_cluster'] = df['title'].map(title_map)
    df = df.dropna(subset=['role_cluster'])
    df['issue_date'] = pd.to_datetime(df['issue_date'])
    df['month'] = df['issue_date'].dt.to_period('M')
    if 'region' not in df.columns:
        df['region'] = 'US'
    df['region'] = df['region'].fillna('US')
    # Merge small regions into "LATAM & Others"
    df['region'] = df['region'].apply(lambda r: r if r in ('US', 'IN') else 'LATAM & Others')
    df['region_role'] = df['region'] + ' | ' + df['role_cluster']

    print(f"  {len(df):,} records, {df['region_role'].nunique()} region|roles")

    # Drop incomplete last month
    monthly_total = df.groupby('month')['openings'].sum().sort_index()
    if len(monthly_total) >= 2:
        last_month = monthly_total.index[-1]
        last_vol = monthly_total.iloc[-1]
        prev_vol = monthly_total.iloc[-2]
        if last_vol < prev_vol * 0.5:
            print(f"  [!] Dropping incomplete {last_month} ({last_vol:,.0f} vs prior {prev_vol:,.0f})")
            df = df[df['month'] != last_month]

    return df, title_map


def load_macro_signals():
    """Load JOLTS, unemployment, yield curve, consumer sentiment."""
    print("  Loading macro signals...")
    jolts = pd.read_csv(os.path.join(DATA_DIR, 'JTSJOL.csv'), parse_dates=['observation_date'])
    jolts['month'] = jolts['observation_date'].dt.to_period('M')
    jolts = jolts.rename(columns={'JTSJOL': 'jolts_openings'})
    jolts['jolts_yoy'] = jolts['jolts_openings'].pct_change(12)
    unrate = pd.read_csv(os.path.join(DATA_DIR, 'UNRATE.csv'), parse_dates=['observation_date'])
    unrate['month'] = unrate['observation_date'].dt.to_period('M')
    spread = pd.read_csv(os.path.join(DATA_DIR, 'T10Y2Y.csv'), parse_dates=['observation_date'])
    spread['month'] = spread['observation_date'].dt.to_period('M')
    spread_m = spread.groupby('month')['T10Y2Y'].mean().reset_index()
    sent = pd.read_csv(os.path.join(DATA_DIR, 'UMCSENT.csv'), parse_dates=['observation_date'])
    sent['month'] = sent['observation_date'].dt.to_period('M')
    print(f"  JOLTS: {jolts['month'].min()} to {jolts['month'].max()}")
    return jolts, unrate, spread_m, sent


def build_panel(df, jolts, unrate, spread_m, sent):
    """Build monthly panel at CLUSTER level with ALL features needed by both models."""
    print("  Building feature panel...")

    ts = df.groupby(['region_role', 'month']).agg(
        openings=('openings', 'sum'),
        n_companies=('company_name', 'nunique'),
    ).reset_index()

    # Viable clusters: >=12 months of data
    cluster_months = ts.groupby('region_role')['month'].nunique()
    viable = cluster_months[cluster_months >= 12].index.tolist()
    panel = ts[ts['region_role'].isin(viable)].copy()

    # ─── Calendar features ───
    panel['cal_month'] = panel['month'].dt.month
    panel['cal_quarter'] = ((panel['cal_month'] - 1) // 3) + 1
    panel['month_sin'] = np.sin(2 * np.pi * panel['cal_month'] / 12)
    panel['month_cos'] = np.cos(2 * np.pi * panel['cal_month'] / 12)
    panel['is_q4'] = (panel['cal_quarter'] == 4).astype(float)
    panel['is_nov_dec'] = panel['cal_month'].isin([11, 12]).astype(float)
    panel['is_q1'] = (panel['cal_quarter'] == 1).astype(float)

    # ─── Exponential decay weights ───
    panel['year'] = panel['month'].dt.year
    weight_map = {2026: 1.0, 2025: 1.0, 2024: 0.7, 2023: 0.5, 2022: 0.3}
    panel['sample_weight'] = panel['year'].map(lambda y: weight_map.get(y, 0.15))

    # ─── Lag & rolling features ───
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

    # ─── Trend (3-month momentum) ───
    panel['trend_3m'] = panel.groupby('region_role')['openings'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).apply(
            lambda s: (s.iloc[-1] - s.iloc[0]) / max(s.iloc[0], 1) if len(s) >= 2 else 0, raw=False))
    panel['trend_3m'] = panel['trend_3m'].fillna(0).clip(-2, 2)

    # ─── YoY change ───
    panel['yoy_change'] = panel.groupby('region_role')['openings'].transform(
        lambda x: (x - x.shift(12)) / x.shift(12).clip(lower=1))
    panel['yoy_change'] = panel['yoy_change'].fillna(0).clip(-2, 2)

    # ─── Seasonal index ───
    cluster_month_avg = panel.groupby(['region_role', 'cal_month'])['openings'].transform('mean')
    cluster_avg = panel.groupby('region_role')['openings'].transform('mean')
    panel['seasonal_idx'] = (cluster_month_avg / cluster_avg.clip(lower=1)).clip(0.2, 3.0)

    # ─── Merge macro signals ───
    panel = panel.merge(jolts[['month', 'jolts_openings', 'jolts_yoy']], on='month', how='left')
    panel = panel.merge(unrate[['month', 'UNRATE']], on='month', how='left')
    panel = panel.merge(spread_m[['month', 'T10Y2Y']], on='month', how='left')
    panel = panel.merge(sent[['month', 'UMCSENT']], on='month', how='left')

    for col in ['jolts_openings', 'jolts_yoy', 'UNRATE', 'T10Y2Y', 'UMCSENT']:
        panel[col] = panel.groupby('region_role')[col].transform(lambda x: x.ffill())

    # Encode cluster index
    le_cluster = LabelEncoder()
    panel['cluster_idx'] = le_cluster.fit_transform(panel['region_role'])
    n_clusters = panel['cluster_idx'].nunique()

    print(f"  Panel: {len(panel):,} rows, {n_clusters} clusters")
    return panel, le_cluster, n_clusters


def cluster_model(cluster_idx, X, y=None, n_clusters=None):
    """Hierarchical Bayesian model — cluster-level random effects."""
    n_feat = X.shape[1]
    mu_global = numpyro.sample('mu_global', dist.Normal(0, 5))
    beta = numpyro.sample('beta', dist.Normal(0, 1).expand([n_feat]))
    sigma_cluster = numpyro.sample('sigma_cluster', dist.HalfNormal(2.0))
    z_cluster = numpyro.sample('z_cluster', dist.Normal(0, 1).expand([n_clusters]))
    alpha_cluster = numpyro.deterministic('alpha_cluster', sigma_cluster * z_cluster)
    sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(3.0))
    mu = mu_global + alpha_cluster[cluster_idx] + X @ beta
    numpyro.sample('obs', dist.Normal(mu, sigma_obs), obs=y)


def fit_single_model(panel_full, feat_cols, label):
    """Fit one model variant. Returns samples, scaler info, and per-cluster MAPE."""
    panel = panel_full.dropna(subset=feat_cols).copy()

    X = panel[feat_cols].values.astype(np.float32)
    y = panel['openings'].values.astype(np.float32)
    weights = panel['sample_weight'].values.astype(np.float32)
    cidx = panel['cluster_idx'].values.astype(np.int32)

    # Standardize
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-6
    X_s = (X - X_mean) / X_std
    y_log = np.log1p(y).astype(np.float32)

    # Weighted resampling
    repeat_map = {1.0: 2, 0.7: 2, 0.5: 1, 0.3: 1, 0.15: 1}
    repeats = np.array([repeat_map.get(w, 1) for w in weights])
    X_w = np.repeat(X_s, repeats, axis=0)
    y_w = np.repeat(y_log, repeats)
    c_w = np.repeat(cidx, repeats)
    nc = panel['cluster_idx'].nunique()

    print(f"    [{label}] {len(y_w):,} rows, {len(feat_cols)} features, {nc} clusters")

    nuts = NUTS(cluster_model, target_accept_prob=0.9, max_tree_depth=10)
    mcmc = MCMC(nuts, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES,
                num_chains=NUM_CHAINS, chain_method='sequential', progress_bar=True)
    mcmc.run(jax.random.PRNGKey(42),
             cluster_idx=jnp.array(c_w), X=jnp.array(X_w), y=jnp.array(y_w),
             n_clusters=nc)
    samples = mcmc.get_samples()

    # In-sample MAPE
    pred_log = (samples['mu_global'][:, None] +
                samples['alpha_cluster'][:, cidx] +
                jnp.dot(samples['beta'], jnp.array(X_s).T))
    fitted = np.expm1(np.median(np.array(pred_log), axis=0))
    ape = np.where(y > 0, np.abs(fitted - y) / y, np.nan)
    ptmp = panel[['region_role']].copy()
    ptmp['_ape'] = ape
    mape_dict = {k: round(v * 100, 1) for k, v in
                 ptmp.groupby('region_role')['_ape'].median().to_dict().items()
                 if not np.isnan(v)}
    med_mape = np.median(list(mape_dict.values()))
    print(f"    [{label}] Median MAPE: {med_mape:.1f}%")

    return {
        'samples': samples,
        'X_mean': X_mean,
        'X_std': X_std,
        'panel': panel,
        'feat_cols': feat_cols,
        'mape': mape_dict,
    }


def predict_single(model, panel_full, forecast_months):
    """Generate predictions from one fitted model."""
    samples = model['samples']
    feat_cols = model['feat_cols']
    X_mean, X_std = model['X_mean'], model['X_std']
    panel = model['panel']

    # Get last known values per cluster
    cluster_last = panel.groupby(['region_role', 'cluster_idx']).last().reset_index()

    # Seasonal index lookup (for model G)
    sidx_lookup = panel.groupby(['region_role', 'cal_month'])['openings'].mean()
    overall_avg = panel.groupby('region_role')['openings'].mean()

    results = []
    for month in forecast_months:
        cal_m = month.month
        cal_q = ((cal_m - 1) // 3) + 1
        cl = cluster_last.copy()

        # Update calendar features for this forecast month
        cl['month_sin'] = np.sin(2 * np.pi * cal_m / 12)
        cl['month_cos'] = np.cos(2 * np.pi * cal_m / 12)
        cl['is_q4'] = float(cal_q == 4)
        cl['is_nov_dec'] = float(cal_m in [11, 12])
        cl['is_q1'] = float(cal_q == 1)

        # Update seasonal index for this calendar month
        if 'seasonal_idx' in feat_cols:
            def get_sidx(rr):
                try:
                    return min(max(sidx_lookup.loc[(rr, cal_m)] / max(overall_avg.loc[rr], 1), 0.2), 3.0)
                except KeyError:
                    return 1.0
            cl['seasonal_idx'] = cl['region_role'].apply(get_sidx)

        X_pred = cl[feat_cols].values.astype(np.float32)
        X_pred_s = (X_pred - X_mean) / X_std
        cpred = cl['cluster_idx'].values.astype(np.int32)

        pred_log = (samples['mu_global'][:, None] +
                    samples['alpha_cluster'][:, cpred] +
                    jnp.dot(samples['beta'], jnp.array(X_pred_s).T))
        pred_arr = np.maximum(0, np.expm1(np.clip(np.array(pred_log), None, 10)))

        for i in range(len(cl)):
            results.append({
                'cluster_name': cl.iloc[i]['region_role'],
                'forecast_month': str(month),
                'predicted': float(np.median(pred_arr[:, i])),
                'draws': pred_arr[:, i],
            })

    return pd.DataFrame(results)


def compute_company_shares(df, n_recent_months=6):
    """Compute each company's share within each cluster using recent N months."""
    print(f"  Computing company shares (last {n_recent_months} months)...")
    last_month = df['month'].max()
    cutoff = last_month - n_recent_months
    recent = df[df['month'] > cutoff]

    cluster_total = recent.groupby('region_role')['openings'].sum()
    company_cluster = recent.groupby(['region_role', 'company_name'])['openings'].sum().reset_index()
    company_cluster['share'] = company_cluster.apply(
        lambda r: r['openings'] / cluster_total.get(r['region_role'], 1), axis=1)

    shares = {}
    for _, row in company_cluster.iterrows():
        rr = row['region_role']
        if rr not in shares:
            shares[rr] = []
        shares[rr].append({
            'company': row['company_name'],
            'share': row['share'],
            'recent_openings': int(row['openings']),
        })

    for rr in shares:
        shares[rr] = sorted(shares[rr], key=lambda x: x['share'], reverse=True)

    print(f"  {len(shares)} clusters with company breakdowns")
    return shares


def write_to_db(forecasts_df, conn, cluster_mape, company_shares, dry_run=False):
    """Write ensemble forecasts to database."""
    if dry_run:
        print(f"\n  DRY RUN — {len(forecasts_df)} rows, {forecasts_df['cluster_name'].nunique()} clusters")
        print(forecasts_df[['cluster_name', 'forecast_month', 'demand_predicted',
                            'demand_lower', 'demand_upper']].head(15).to_string(index=False))
        return

    cur = conn.cursor(dictionary=True)
    clusters = forecasts_df['cluster_name'].unique()

    # Get existing metadata
    if len(clusters) > 0:
        ph = ','.join(['%s'] * len(clusters))
        cur.execute(f"""SELECT DISTINCT cluster_name, top_skills, top_locations, top_clients
                        FROM demand_forecasts WHERE cluster_name IN ({ph})""", list(clusters))
        meta = {r['cluster_name']: r for r in cur.fetchall()}
    else:
        meta = {}

    # Delete old predictions for these months
    for m in forecasts_df['forecast_month'].unique():
        cur.execute("DELETE FROM demand_forecasts WHERE forecast_month=%s AND data_type='predicted'", (m,))

    insert_sql = """
        INSERT INTO demand_forecasts
        (cluster_name, forecast_month, demand_predicted, demand_lower, demand_upper,
         model_used, mape, mae, mase, is_reliable, top_skills, top_locations, top_clients, data_type)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'predicted')
    """
    rows = 0
    for _, row in forecasts_df.iterrows():
        m = meta.get(row['cluster_name'], {})
        mape = cluster_mape.get(row['cluster_name'])
        is_reliable = 1 if (mape is not None and mape <= 50) else 0
        shares = company_shares.get(row['cluster_name'], [])
        top_clients = json.dumps([s['company'] for s in shares[:5]]) if shares else m.get('top_clients')
        cur.execute(insert_sql, (
            row['cluster_name'], row['forecast_month'],
            int(row['demand_predicted']), int(row['demand_lower']), int(row['demand_upper']),
            MODEL_VERSION, mape, None, None, is_reliable,
            m.get('top_skills'), m.get('top_locations'), top_clients,
        ))
        rows += 1

    # Update company_cluster_profiles
    cur.execute("TRUNCATE TABLE company_cluster_profiles")
    profiles = 0
    for cluster, companies in company_shares.items():
        for comp in companies:
            cur.execute("""INSERT INTO company_cluster_profiles
                           (cluster_name, company_name, jd_count, openings, top_skills, top_locations)
                           VALUES (%s,%s,%s,%s,NULL,NULL)""",
                        (cluster, comp['company'], comp['recent_openings'], comp['recent_openings']))
            profiles += 1

    conn.commit()
    cur.close()
    print(f"\n  [OK] Inserted {rows} forecast rows for {len(clusters)} clusters")
    print(f"  [OK] Updated {profiles} company_cluster_profiles")


def main():
    parser = argparse.ArgumentParser(description='Generate ensemble demand forecasts')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--months', type=int, default=6)
    args = parser.parse_args()

    print("=" * 70)
    print("  ENSEMBLE FORECAST — Max(A, G)")
    print(f"  Model: {MODEL_VERSION}")
    print(f"  A = Lag model | G = Seasonal model | Ensemble = Max(A,G)")
    print("=" * 70)

    # Load data
    df, title_map = load_data()
    jolts, unrate, spread_m, sent = load_macro_signals()
    panel, le_cluster, n_clusters = build_panel(df, jolts, unrate, spread_m, sent)
    company_shares = compute_company_shares(df)

    # Forecast months
    last_month = panel.dropna(subset=FEAT_A)['month'].max()
    forecast_months = pd.period_range(start=last_month + 1, periods=args.months, freq='M')
    print(f"\n  Forecasting: {forecast_months[0]} -> {forecast_months[-1]}")

    # ─── Fit Model A (Lag) ───
    print("\n  === Model A (Lag) ===")
    model_a = fit_single_model(panel, FEAT_A, "A")

    # ─── Fit Model G (Seasonal) ───
    print("\n  === Model G (Seasonal) ===")
    model_g = fit_single_model(panel, FEAT_G, "G")

    # ─── Predict from both ───
    pred_a = predict_single(model_a, panel, forecast_months)
    pred_g = predict_single(model_g, panel, forecast_months)

    # ─── Ensemble: Max(A, G) ───
    print("\n  Ensembling: Max(A, G)...")
    merged = pred_a.merge(pred_g, on=['cluster_name', 'forecast_month'], suffixes=('_A', '_G'))

    forecasts = []
    for _, row in merged.iterrows():
        pred_max = max(row['predicted_A'], row['predicted_G'])
        # Combine draws from both models for honest CI
        all_draws = np.concatenate([row['draws_A'], row['draws_G']])
        lo = float(np.percentile(all_draws, CI_LO_PCT))
        hi = float(np.percentile(all_draws, CI_HI_PCT))
        forecasts.append({
            'cluster_name': row['cluster_name'],
            'forecast_month': row['forecast_month'],
            'demand_predicted': max(0, round(pred_max)),
            'demand_lower': max(0, round(lo)),
            'demand_upper': max(1, round(hi)),
        })

    forecasts_df = pd.DataFrame(forecasts)

    # Best MAPE from either model per cluster
    cluster_mape = {}
    all_clusters = set(list(model_a['mape'].keys()) + list(model_g['mape'].keys()))
    for k in all_clusters:
        cluster_mape[k] = min(model_a['mape'].get(k, 999), model_g['mape'].get(k, 999))

    # ─── Summary ───
    total = forecasts_df['demand_predicted'].sum()
    total_lo = forecasts_df['demand_lower'].sum()
    total_hi = forecasts_df['demand_upper'].sum()
    n_out = forecasts_df['cluster_name'].nunique()
    print(f"\n  Forecasts: {len(forecasts_df)} rows, {n_out} clusters")
    print(f"  Total H2: {total_lo:,} — {total:,} — {total_hi:,}")
    print(f"  Model A alone: {pred_a['predicted'].sum():,.0f}")
    print(f"  Model G alone: {pred_g['predicted'].sum():,.0f}")
    print(f"  Ensemble Max(A,G): {total:,}")

    # ─── Write to DB ───
    out = forecasts_df[['cluster_name', 'forecast_month', 'demand_predicted',
                        'demand_lower', 'demand_upper']]
    if not args.dry_run:
        conn = mysql.connector.connect(**DB_CONFIG)
        write_to_db(out, conn, cluster_mape, company_shares, dry_run=False)
        conn.close()
    else:
        write_to_db(out, None, cluster_mape, company_shares, dry_run=True)

    print(f"\n  Model: {MODEL_VERSION}")
    print("  Done!")


if __name__ == '__main__':
    main()
