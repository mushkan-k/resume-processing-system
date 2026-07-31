"""
Generate demand forecasts using Hierarchical Bayesian model + JOLTS macro signals.
===================================================================================
Two-stage approach (Option C):
  Stage 1: Forecast at CLUSTER level (region_role) — clean signal, no company noise
  Stage 2: Split by company using recent proportions (last 6 months)

Key improvements over combo-level:
  - Company spikes don't inflate cluster forecasts
  - More data per time series = better pattern learning
  - Seasonal patterns are clearer at cluster level
  - Company shares reflect CURRENT reality (last 6 months), not historical spikes

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

MODEL_VERSION = "HierBayes_ClusterLevel"
NUM_WARMUP = 1000
NUM_SAMPLES = 2000
NUM_CHAINS = 2
CI_LO_PCT = 20
CI_HI_PCT = 80


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
    # Merge small regions into "LATAM & Others" — keep US and IN separate
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


def build_cluster_panel(df, jolts, unrate, spread_m, sent):
    """
    Stage 1 data prep: Build monthly panel at CLUSTER level (not combo).
    Each row = one region_role x one month.
    """
    print("  Building cluster-level panel...")

    # Aggregate to cluster x month (NO company dimension)
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

    # ─── Exponential decay weights ───
    panel['year'] = panel['month'].dt.year
    weight_map = {2026: 1.0, 2025: 1.0, 2024: 0.7, 2023: 0.5, 2022: 0.3}
    panel['sample_weight'] = panel['year'].map(lambda y: weight_map.get(y, 0.15))

    # ─── Lag features (at cluster level — much smoother than combo level) ───
    panel = panel.sort_values(['region_role', 'month']).reset_index(drop=True)
    panel['lag1'] = panel.groupby('region_role')['openings'].shift(1)
    panel['lag2'] = panel.groupby('region_role')['openings'].shift(2)
    panel['lag3'] = panel.groupby('region_role')['openings'].shift(3)
    panel['roll3'] = panel.groupby('region_role')['openings'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    panel['roll6'] = panel.groupby('region_role')['openings'].transform(
        lambda x: x.shift(1).rolling(6, min_periods=3).mean())

    # ─── Trend (3-month momentum) ───
    panel['trend_3m'] = panel.groupby('region_role')['openings'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=2).apply(
            lambda s: (s.iloc[-1] - s.iloc[0]) / max(s.iloc[0], 1) if len(s) >= 2 else 0, raw=False))
    panel['trend_3m'] = panel['trend_3m'].fillna(0).clip(-2, 2)

    # ─── YoY change (same month last year) ───
    panel['yoy_change'] = panel.groupby('region_role')['openings'].transform(
        lambda x: (x - x.shift(12)) / x.shift(12).clip(lower=1))
    panel['yoy_change'] = panel['yoy_change'].fillna(0).clip(-2, 2)

    # Merge macro signals
    panel = panel.merge(jolts[['month', 'jolts_openings', 'jolts_yoy']], on='month', how='left')
    panel = panel.merge(unrate[['month', 'UNRATE']], on='month', how='left')
    panel = panel.merge(spread_m[['month', 'T10Y2Y']], on='month', how='left')
    panel = panel.merge(sent[['month', 'UMCSENT']], on='month', how='left')

    for col in ['jolts_openings', 'jolts_yoy', 'UNRATE', 'T10Y2Y', 'UMCSENT']:
        panel[col] = panel.groupby('region_role')[col].transform(lambda x: x.ffill())

    panel = panel.dropna(subset=['lag1', 'lag2', 'lag3', 'roll3', 'roll6',
                                  'jolts_openings', 'UNRATE', 'T10Y2Y', 'UMCSENT']).copy()

    # Encode cluster index
    le_cluster = LabelEncoder()
    panel['cluster_idx'] = le_cluster.fit_transform(panel['region_role'])
    n_clusters = panel['cluster_idx'].nunique()

    print(f"  Panel: {len(panel):,} rows, {n_clusters} clusters")
    print(f"  Weight distribution: {panel['sample_weight'].value_counts().sort_index().to_dict()}")

    return panel, le_cluster, n_clusters


def compute_company_shares(df, n_recent_months=6):
    """
    Stage 2: Compute each company's share within each cluster
    using only the MOST RECENT N months.
    """
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


def cluster_model(cluster_idx, X, y=None, n_clusters=None):
    """Simpler hierarchical model — only cluster-level random effects.
    No company/combo effects = no memorized company spikes."""
    n_feat = X.shape[1]
    mu_global = numpyro.sample('mu_global', dist.Normal(0, 5))
    beta = numpyro.sample('beta', dist.Normal(0, 1).expand([n_feat]))
    sigma_cluster = numpyro.sample('sigma_cluster', dist.HalfNormal(2.0))
    z_cluster = numpyro.sample('z_cluster', dist.Normal(0, 1).expand([n_clusters]))
    alpha_cluster = numpyro.deterministic('alpha_cluster', sigma_cluster * z_cluster)
    sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(3.0))
    mu = mu_global + alpha_cluster[cluster_idx] + X @ beta
    numpyro.sample('obs', dist.Normal(mu, sigma_obs), obs=y)


def fit_and_predict(panel, n_clusters, forecast_months):
    """Fit cluster-level model and generate forecasts."""

    feat_cols = ['lag1', 'lag2', 'lag3', 'roll3', 'roll6', 'trend_3m', 'yoy_change',
                 'month_sin', 'month_cos', 'is_q4', 'is_nov_dec',
                 'jolts_openings', 'jolts_yoy', 'UNRATE', 'T10Y2Y', 'UMCSENT']

    X_train = panel[feat_cols].values.astype(np.float32)
    y_train = panel['openings'].values.astype(np.float32)
    weights = panel['sample_weight'].values.astype(np.float32)
    cluster_train = panel['cluster_idx'].values.astype(np.int32)

    # Standardize
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0) + 1e-6
    X_train_s = (X_train - X_mean) / X_std
    y_train_log = np.log1p(y_train).astype(np.float32)

    # Weighted resampling
    repeat_map = {1.0: 2, 0.7: 2, 0.5: 1, 0.3: 1, 0.15: 1}
    repeats = np.array([repeat_map.get(w, 1) for w in weights])
    X_w = np.repeat(X_train_s, repeats, axis=0)
    y_w = np.repeat(y_train_log, repeats)
    cluster_w = np.repeat(cluster_train, repeats)

    print(f"\n  Training rows (weighted): {len(y_w):,} (original: {len(y_train):,})")
    print(f"  Features: {len(feat_cols)}")
    print(f"  Fitting cluster-level model ({NUM_CHAINS} chains x {NUM_SAMPLES} samples)...")

    nuts = NUTS(cluster_model, target_accept_prob=0.9, max_tree_depth=10)
    mcmc = MCMC(nuts, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES,
                num_chains=NUM_CHAINS, chain_method='sequential', progress_bar=True)
    mcmc.run(jax.random.PRNGKey(42),
             cluster_idx=jnp.array(cluster_w),
             X=jnp.array(X_w), y=jnp.array(y_w),
             n_clusters=n_clusters)

    samples = mcmc.get_samples()

    # ─── In-sample MAPE ───
    print("  Computing per-cluster accuracy...")
    pred_log_train = (samples['mu_global'][:, None] +
                      samples['alpha_cluster'][:, cluster_train] +
                      jnp.dot(samples['beta'], jnp.array(X_train_s).T))
    fitted = np.expm1(np.median(np.array(pred_log_train), axis=0))
    ape = np.where(y_train > 0, np.abs(fitted - y_train) / y_train, np.nan)
    panel_tmp = panel[['region_role']].copy()
    panel_tmp['_ape'] = ape
    cluster_mape = panel_tmp.groupby('region_role')['_ape'].median().to_dict()
    cluster_mape = {k: round(v * 100, 1) for k, v in cluster_mape.items() if not np.isnan(v)}
    print(f"  Per-cluster MAPE: {len(cluster_mape)} clusters, median={np.median(list(cluster_mape.values())):.1f}%")

    # ─── Generate forecasts ───
    print("  Generating forecasts...")
    cluster_last = panel.groupby(['region_role', 'cluster_idx']).agg(
        lag1=('lag1', 'last'), lag2=('lag2', 'last'), lag3=('lag3', 'last'),
        roll3=('roll3', 'last'), roll6=('roll6', 'last'),
        trend_3m=('trend_3m', 'last'), yoy_change=('yoy_change', 'last'),
        jolts_openings=('jolts_openings', 'last'), jolts_yoy=('jolts_yoy', 'last'),
        UNRATE=('UNRATE', 'last'), T10Y2Y=('T10Y2Y', 'last'), UMCSENT=('UMCSENT', 'last'),
    ).reset_index()

    forecasts = []
    for month in forecast_months:
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
        lo = np.percentile(pred_arr, CI_LO_PCT, axis=0)
        hi = np.percentile(pred_arr, CI_HI_PCT, axis=0)

        for i in range(len(cl)):
            forecasts.append({
                'cluster_name': cl.iloc[i]['region_role'],
                'forecast_month': str(month),
                'demand_predicted': max(0, round(float(point[i]))),
                'demand_lower': max(0, round(float(lo[i]))),
                'demand_upper': max(1, round(float(hi[i]))),
                '_draws': pred_arr[:, i],
            })

    return pd.DataFrame(forecasts), cluster_mape


def get_cluster_metadata(conn, cluster_names):
    cur = conn.cursor(dictionary=True)
    placeholders = ','.join(['%s'] * len(cluster_names))
    cur.execute(f"""
        SELECT DISTINCT cluster_name, top_skills, top_locations, top_clients
        FROM demand_forecasts WHERE cluster_name IN ({placeholders})
    """, list(cluster_names))
    meta = {r['cluster_name']: r for r in cur.fetchall()}
    cur.close()
    return meta


def write_to_db(forecasts_df, conn, cluster_mape, company_shares, dry_run=False):
    if dry_run:
        print(f"\n  DRY RUN — {len(forecasts_df)} rows, {forecasts_df['cluster_name'].nunique()} clusters")
        print(forecasts_df[['cluster_name','forecast_month','demand_predicted']].head(12).to_string(index=False))
        return

    cur = conn.cursor(dictionary=True)
    clusters = forecasts_df['cluster_name'].unique()
    meta = get_cluster_metadata(conn, clusters)

    months = forecasts_df['forecast_month'].unique()
    for m in months:
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

    # Update company_cluster_profiles with recent shares
    cur.execute("TRUNCATE TABLE company_cluster_profiles")
    profiles = 0
    for cluster, companies in company_shares.items():
        for comp in companies:
            cur.execute("""
                INSERT INTO company_cluster_profiles (cluster_name, company_name, jd_count, openings, top_skills, top_locations)
                VALUES (%s,%s,%s,%s,NULL,NULL)
            """, (cluster, comp['company'], comp['recent_openings'], comp['recent_openings']))
            profiles += 1

    conn.commit()
    cur.close()
    print(f"\n  [OK] Inserted {rows} forecast rows for {len(clusters)} clusters")
    print(f"  [OK] Updated {profiles} company_cluster_profiles (recent 6-month shares)")


def main():
    parser = argparse.ArgumentParser(description='Generate demand forecasts (cluster-level)')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--months', type=int, default=6)
    args = parser.parse_args()

    print("=" * 70)
    print("  HIERARCHICAL BAYESIAN DEMAND FORECAST — CLUSTER LEVEL (v2)")
    print(f"  Model: {MODEL_VERSION}")
    print(f"  Stage 1: Forecast at cluster level (no company noise)")
    print(f"  Stage 2: Company shares from last 6 months")
    print("=" * 70)

    df, title_map = load_data()
    jolts, unrate, spread_m, sent = load_macro_signals()
    panel, le_cluster, n_clusters = build_cluster_panel(df, jolts, unrate, spread_m, sent)
    company_shares = compute_company_shares(df)

    last_month = panel['month'].max()
    forecast_months = pd.period_range(start=last_month + 1, periods=args.months, freq='M')
    print(f"\n  Forecasting: {forecast_months[0]} -> {forecast_months[-1]}")

    forecasts_df, cluster_mape = fit_and_predict(panel, n_clusters, forecast_months)

    total = forecasts_df['demand_predicted'].sum()
    n_out = forecasts_df['cluster_name'].nunique()
    print(f"\n  Forecasts: {len(forecasts_df)} rows, {n_out} clusters")
    print(f"  Total H2 forecast: {total:,} openings")

    all_draws = np.array(forecasts_df['_draws'].tolist())
    port_draws = all_draws.sum(axis=0)
    print(f"  Portfolio: {int(np.percentile(port_draws, CI_LO_PCT)):,} — {int(np.median(port_draws)):,} — {int(np.percentile(port_draws, CI_HI_PCT)):,}")

    out = forecasts_df[['cluster_name', 'forecast_month', 'demand_predicted', 'demand_lower', 'demand_upper']]
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
