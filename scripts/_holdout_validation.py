"""
Holdout Validation: Train on data through April 2026, predict May-Jun, compare to actuals.
This gives us the REAL out-of-sample accuracy.
"""
import os, sys, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from sklearn.preprocessing import LabelEncoder

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# Load data (same as main script)
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
df['region_role'] = df['region'] + ' | ' + df['role_cluster']

# Monthly aggregation
ts = df.groupby(['region_role', 'company_name', 'month']).agg(openings=('openings', 'sum')).reset_index()
ts['combo'] = list(zip(ts['region_role'], ts['company_name']))

# Viable combos (18+ months)
combo_months = ts.groupby('combo')['month'].nunique()
viable = combo_months[combo_months >= 18].index.tolist()
panel = ts[ts['combo'].isin(viable)].copy()

# Add features
panel['year'] = panel['month'].dt.year
panel['cal_month'] = panel['month'].dt.month
panel['month_sin'] = np.sin(2 * np.pi * panel['cal_month'] / 12)
panel['month_cos'] = np.cos(2 * np.pi * panel['cal_month'] / 12)
panel['is_q4'] = (((panel['cal_month'] - 1) // 3 + 1) == 4).astype(float)
panel['is_nov_dec'] = panel['cal_month'].isin([11, 12]).astype(float)

# Weights
weight_map = {2026: 1.0, 2025: 1.0, 2024: 0.7, 2023: 0.5, 2022: 0.3}
panel['sample_weight'] = panel['year'].map(lambda y: weight_map.get(y, 0.15))

# Lags & trend
panel = panel.sort_values(['region_role', 'company_name', 'month']).reset_index(drop=True)
panel['lag1'] = panel.groupby('combo')['openings'].shift(1)
panel['lag2'] = panel.groupby('combo')['openings'].shift(2)
panel['lag3'] = panel.groupby('combo')['openings'].shift(3)
panel['roll3'] = panel.groupby('combo')['openings'].transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
panel['trend_3m'] = panel.groupby('combo')['openings'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=2).apply(
        lambda s: (s.iloc[-1] - s.iloc[0]) / max(s.iloc[0], 1) if len(s) >= 2 else 0, raw=False))
panel['trend_3m'] = panel['trend_3m'].fillna(0).clip(-2, 2)

# Load macro
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

panel = panel.merge(jolts[['month', 'jolts_openings', 'jolts_yoy']], on='month', how='left')
panel = panel.merge(unrate[['month', 'UNRATE']], on='month', how='left')
panel = panel.merge(spread_m[['month', 'T10Y2Y']], on='month', how='left')
panel = panel.merge(sent[['month', 'UMCSENT']], on='month', how='left')
for col in ['jolts_openings', 'jolts_yoy', 'UNRATE', 'T10Y2Y', 'UMCSENT']:
    panel[col] = panel.groupby('combo')[col].transform(lambda x: x.ffill())

panel = panel.dropna(subset=['lag1', 'lag2', 'lag3', 'roll3', 'jolts_openings', 'UNRATE', 'T10Y2Y', 'UMCSENT']).copy()

# Split: Train <= Apr 2026, Test = May-Jun 2026
train_cutoff = pd.Period('2026-04', freq='M')
train = panel[panel['month'] <= train_cutoff].copy()
test = panel[panel['month'] > train_cutoff].copy()

print(f"Train: {len(train)} rows (through {train_cutoff})")
print(f"Test: {len(test)} rows ({test['month'].min()} to {test['month'].max()})")

# Encode
le_role = LabelEncoder()
le_comp = LabelEncoder()
le_combo = LabelEncoder()
train['role_idx'] = le_role.fit_transform(train['region_role'])
train['comp_idx'] = le_comp.fit_transform(train['company_name'])
train['combo_idx'] = le_combo.fit_transform(train['combo'].astype(str))
n_roles = train['role_idx'].nunique()
n_comps = train['comp_idx'].nunique()
n_combos = train['combo_idx'].nunique()

feat_cols = ['lag1', 'lag2', 'lag3', 'roll3', 'trend_3m',
             'month_sin', 'month_cos', 'is_q4', 'is_nov_dec',
             'jolts_openings', 'jolts_yoy', 'UNRATE', 'T10Y2Y', 'UMCSENT']

X_train = train[feat_cols].values.astype(np.float32)
y_train = train['openings'].values.astype(np.float32)
weights = train['sample_weight'].values.astype(np.float32)

# Weighting
repeat_map = {1.0: 2, 0.7: 2, 0.5: 1, 0.3: 1, 0.15: 1}
repeats = np.array([repeat_map.get(w, 1) for w in weights])
X_train_w = np.repeat(X_train, repeats, axis=0)
y_train_w = np.repeat(np.log1p(y_train), repeats)
role_w = np.repeat(train['role_idx'].values, repeats)
comp_w = np.repeat(train['comp_idx'].values, repeats)
combo_w = np.repeat(train['combo_idx'].values, repeats)

X_mean = X_train.mean(axis=0)
X_std = X_train.std(axis=0) + 1e-6
X_train_s = (X_train_w - X_mean) / X_std

print(f"Training (weighted): {len(X_train_s)} rows")

# Model (same as main)
def hier_model(role_idx, comp_idx, combo_idx, X, y=None, n_roles=None, n_comps=None, n_combos=None):
    n_feat = X.shape[1]
    mu_global = numpyro.sample('mu_global', dist.Normal(0, 5))
    beta = numpyro.sample('beta', dist.Normal(0, 1).expand([n_feat]))
    sigma_role = numpyro.sample('sigma_role', dist.HalfNormal(2.0))
    z_role = numpyro.sample('z_role', dist.Normal(0, 1).expand([n_roles]))
    alpha_role = numpyro.deterministic('alpha_role', sigma_role * z_role)
    sigma_comp = numpyro.sample('sigma_comp', dist.HalfNormal(2.0))
    z_comp = numpyro.sample('z_comp', dist.Normal(0, 1).expand([n_comps]))
    alpha_comp = numpyro.deterministic('alpha_comp', sigma_comp * z_comp)
    sigma_combo = numpyro.sample('sigma_combo', dist.HalfNormal(2.0))
    z_combo = numpyro.sample('z_combo', dist.Normal(0, 1).expand([n_combos]))
    alpha_combo = numpyro.deterministic('alpha_combo', sigma_combo * z_combo)
    sigma_obs = numpyro.sample('sigma_obs', dist.HalfNormal(3.0))
    mu = mu_global + alpha_role[role_idx] + alpha_comp[comp_idx] + alpha_combo[combo_idx] + X @ beta
    numpyro.sample('obs', dist.Normal(mu, sigma_obs), obs=y)

# Fit
nuts = NUTS(hier_model, target_accept_prob=0.9, max_tree_depth=10)
mcmc = MCMC(nuts, num_warmup=800, num_samples=1500, num_chains=2, chain_method='sequential', progress_bar=True)
mcmc.run(jax.random.PRNGKey(42),
         role_idx=jnp.array(role_w.astype(np.int32)),
         comp_idx=jnp.array(comp_w.astype(np.int32)),
         combo_idx=jnp.array(combo_w.astype(np.int32)),
         X=jnp.array(X_train_s), y=jnp.array(y_train_w.astype(np.float32)),
         n_roles=n_roles, n_comps=n_comps, n_combos=n_combos)

# Predict on test set
samples = mcmc.get_samples()

# Map test rows to train's label encoders
test_valid = test[test['region_role'].isin(le_role.classes_) & 
                  test['company_name'].isin(le_comp.classes_)].copy()
test_valid['role_idx'] = le_role.transform(test_valid['region_role'])
test_valid['comp_idx'] = le_comp.transform(test_valid['company_name'])
test_valid['combo_str'] = test_valid['combo'].astype(str)
test_valid = test_valid[test_valid['combo_str'].isin(le_combo.classes_)]
test_valid['combo_idx'] = le_combo.transform(test_valid['combo_str'])

X_test = test_valid[feat_cols].values.astype(np.float32)
X_test_s = (X_test - X_mean) / X_std

pred_log = (samples['mu_global'][:, None] +
            samples['alpha_role'][:, test_valid['role_idx'].values] +
            samples['alpha_comp'][:, test_valid['comp_idx'].values] +
            samples['alpha_combo'][:, test_valid['combo_idx'].values] +
            jnp.dot(samples['beta'], jnp.array(X_test_s).T))

pred = np.expm1(np.median(np.array(pred_log), axis=0))
actuals = test_valid['openings'].values

# MAPE
mask = actuals > 0
ape = np.abs(pred[mask] - actuals[mask]) / actuals[mask]

print(f"\n{'='*60}")
print(f"HOLDOUT VALIDATION (May-Jun 2026)")
print(f"{'='*60}")
print(f"Test rows: {len(test_valid)} (combos that existed in training)")
print(f"Mean APE: {np.mean(ape)*100:.1f}%")
print(f"Median APE: {np.median(ape)*100:.1f}%")
print(f"Predicted total: {int(pred.sum()):,}")
print(f"Actual total: {int(actuals.sum()):,}")
print(f"Ratio: {pred.sum()/actuals.sum():.2f}x")

# Per-cluster holdout MAPE
test_valid['pred'] = pred
test_valid['actual'] = actuals
cluster_results = test_valid.groupby('region_role').agg(
    pred_sum=('pred', 'sum'),
    actual_sum=('actual', 'sum'),
).reset_index()
cluster_results['ape'] = np.abs(cluster_results['pred_sum'] - cluster_results['actual_sum']) / cluster_results['actual_sum'].clip(lower=1)
cluster_results = cluster_results.sort_values('actual_sum', ascending=False)

print(f"\nTop 15 clusters (by actual demand):")
print(f"{'Cluster':<35} {'Pred':>5} {'Actual':>6} {'APE':>6}")
print("-" * 55)
for _, r in cluster_results.head(15).iterrows():
    print(f"{r['region_role']:<35} {int(r['pred_sum']):>5} {int(r['actual_sum']):>6} {r['ape']*100:>5.1f}%")

# WMAPE on holdout
total_actual = cluster_results['actual_sum'].sum()
wmape = (cluster_results['ape'] * cluster_results['actual_sum'] / total_actual).sum()
print(f"\nHoldout WMAPE: {wmape*100:.1f}%")
print(f"Holdout Accuracy (100-WMAPE): {(1-wmape)*100:.1f}%")
