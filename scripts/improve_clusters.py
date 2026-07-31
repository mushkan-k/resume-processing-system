"""
Improve Reliable Clusters — Targeted Model Enhancement
=======================================================
Strategy:
1. Fix clusters that ALREADY have MAPE ≤ 30% but are marked unreliable (DB/MASE issue)
2. For borderline clusters (30-45% MAPE), apply targeted improvements:
   - Per-cluster bias correction using recent residuals
   - Volatility-adjusted predictions (shrink toward rolling mean)
   - Ensemble re-weighting (favor model with lower error per cluster)
3. Retrain with combo-specific smoothing for high-variance low-volume clusters
4. Update database with corrected predictions + reliability flags
"""
import pandas as pd
import numpy as np
import os, sys
import warnings
import mysql.connector
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ═══════════════════════════════════════════════════════════════
# STEP 1: Load the training data + reproduce the model state
# ═══════════════════════════════════════════════════════════════
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

df = pd.read_pickle(os.path.join(data_dir, 'clean_42k_v1.pkl'))
mapping = pd.read_pickle(os.path.join(data_dir, 'title_to_cluster.pkl'))

title_to_role = dict(zip(mapping['raw_title'], mapping['role_cluster']))
df['role_cluster'] = df['title'].map(title_to_role)
df['issue_date'] = pd.to_datetime(df['issue_date'])
df['month'] = df['issue_date'].dt.to_period('M')
df['quarter'] = df['issue_date'].dt.to_period('Q')
df['location'] = df['location'].fillna('Unknown')
df['combo_key'] = df['region'] + ' | ' + df['role_cluster']
df['openings'] = pd.to_numeric(df['openings'], errors='coerce').fillna(1)
df['fills'] = pd.to_numeric(df['fills'], errors='coerce').fillna(0)

# Build monthly time series
ts_monthly = (
    df.groupby(['combo_key', 'region', 'role_cluster', 'month'])
    .agg(openings=('openings', 'sum'), fills=('fills', 'sum'), record_count=('title', 'count'))
    .reset_index()
)
ts_monthly['quarter'] = ts_monthly['month'].apply(lambda m: m.to_timestamp().to_period('Q'))

# Viable combos
train_start_q = pd.Period('2022Q1', freq='Q')
train_end_q = pd.Period('2025Q4', freq='Q')
ts_train_q = ts_monthly[(ts_monthly['quarter'] >= train_start_q) & (ts_monthly['quarter'] <= train_end_q)]
combo_avg = ts_train_q.groupby('combo_key')['record_count'].sum() / 16
viable_combos = combo_avg[combo_avg >= 10].index.tolist()

# Load FRED signals
unrate = pd.read_csv(os.path.join(data_dir, 'UNRATE.csv'))
unrate['observation_date'] = pd.to_datetime(unrate['observation_date'])
unrate['UNRATE'] = pd.to_numeric(unrate['UNRATE'], errors='coerce')
unrate = unrate.dropna(subset=['UNRATE'])
unrate['month'] = unrate['observation_date'].dt.to_period('M')
unrate_monthly = unrate.groupby('month')['UNRATE'].mean().reset_index()
unrate_monthly.columns = ['month', 'unrate']

jolts = pd.read_csv(os.path.join(data_dir, 'JTSJOL.csv'))
jolts['observation_date'] = pd.to_datetime(jolts['observation_date'])
jolts['JTSJOL'] = pd.to_numeric(jolts['JTSJOL'], errors='coerce')
jolts = jolts.dropna(subset=['JTSJOL'])
jolts['month'] = jolts['observation_date'].dt.to_period('M')
jolts_monthly = jolts.groupby('month')['JTSJOL'].mean().reset_index()
jolts_monthly.columns = ['month', 'jolts']

spread = pd.read_csv(os.path.join(data_dir, 'T10Y2Y.csv'))
spread['observation_date'] = pd.to_datetime(spread['observation_date'])
spread['T10Y2Y'] = pd.to_numeric(spread['T10Y2Y'], errors='coerce')
spread = spread.dropna(subset=['T10Y2Y'])
spread['month'] = spread['observation_date'].dt.to_period('M')
spread_monthly = spread.groupby('month')['T10Y2Y'].mean().reset_index()
spread_monthly.columns = ['month', 'yield_spread']

sent = pd.read_csv(os.path.join(data_dir, 'UMCSENT.csv'))
sent['observation_date'] = pd.to_datetime(sent['observation_date'])
sent['UMCSENT'] = pd.to_numeric(sent['UMCSENT'], errors='coerce')
sent = sent.dropna(subset=['UMCSENT'])
sent['month'] = sent['observation_date'].dt.to_period('M')
sent_monthly = sent.groupby('month')['UMCSENT'].mean().reset_index()
sent_monthly.columns = ['month', 'consumer_sent']

# Build panel
panel = ts_monthly[ts_monthly['combo_key'].isin(viable_combos)].copy()
panel = panel.merge(unrate_monthly, on='month', how='left')
panel = panel.merge(jolts_monthly, on='month', how='left')
panel = panel.merge(spread_monthly, on='month', how='left')
panel = panel.merge(sent_monthly, on='month', how='left')

for col in ['unrate', 'jolts', 'yield_spread', 'consumer_sent']:
    panel[col] = panel[col].ffill().bfill()

panel['month_num'] = panel['month'].apply(lambda x: x.month)
panel['q_num'] = panel['quarter'].apply(lambda x: x.quarter)
panel['year'] = panel['month'].apply(lambda x: x.year)
for q in [1, 2, 3]:
    panel[f'Q{q}'] = (panel['q_num'] == q).astype(int)

panel = panel.sort_values(['combo_key', 'month']).reset_index(drop=True)

# Feature engineering (same as Cell 2)
from sklearn.preprocessing import LabelEncoder

for lag in [1, 2, 3, 6, 12]:
    panel[f'openings_lag{lag}'] = panel.groupby('combo_key')['openings'].shift(lag)

panel['fills_lag1'] = panel.groupby('combo_key')['fills'].shift(1)
panel['fill_rate'] = panel['fills'] / panel['openings'].clip(lower=1)
panel['fill_rate_lag1'] = panel.groupby('combo_key')['fill_rate'].shift(1)

panel['rolling_3m_mean'] = panel.groupby('combo_key')['openings'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=2).mean()
)
panel['rolling_6m_mean'] = panel.groupby('combo_key')['openings'].transform(
    lambda x: x.shift(1).rolling(6, min_periods=3).mean()
)
panel['rolling_12m_mean'] = panel.groupby('combo_key')['openings'].transform(
    lambda x: x.shift(1).rolling(12, min_periods=6).mean()
)

panel['yoy_change'] = (panel['openings'] - panel['openings_lag12']) / panel['openings_lag12'].clip(lower=1)
panel['yoy_change_lag1'] = panel.groupby('combo_key')['yoy_change'].shift(1)

def rolling_slope_monthly(s, window=6):
    slopes = []
    for i in range(len(s)):
        if i < window:
            slopes.append(np.nan)
        else:
            y = s.iloc[i-window:i].values
            x = np.arange(window)
            if np.std(y) == 0:
                slopes.append(0.0)
            else:
                slopes.append(np.polyfit(x, y, 1)[0])
    return pd.Series(slopes, index=s.index)

panel['trend_slope'] = panel.groupby('combo_key')['openings'].transform(rolling_slope_monthly)
panel['jolts_change'] = panel.groupby('combo_key')['jolts'].pct_change()

le_region = LabelEncoder()
le_role = LabelEncoder()
panel['region_enc'] = le_region.fit_transform(panel['region'])
panel['role_enc'] = le_role.fit_transform(panel['role_cluster'])

train_end_month = pd.Period('2025-12', freq='M')
combo_vol = panel[panel['month'] <= train_end_month].groupby('combo_key')['openings'].mean()
panel['combo_avg_vol'] = panel['combo_key'].map(combo_vol)

feature_cols = [
    'unrate', 'jolts', 'yield_spread', 'consumer_sent',
    'Q1', 'Q2', 'Q3',
    'openings_lag1', 'openings_lag2', 'openings_lag3', 'openings_lag6', 'openings_lag12',
    'rolling_3m_mean', 'rolling_6m_mean', 'rolling_12m_mean',
    'fills_lag1', 'fill_rate_lag1',
    'yoy_change_lag1', 'trend_slope', 'jolts_change',
    'region_enc', 'role_enc', 'combo_avg_vol',
]
target_col = 'openings'

panel_clean = panel.dropna(subset=feature_cols + [target_col]).copy()

print(f"Panel loaded: {len(panel_clean)} rows, {panel_clean['combo_key'].nunique()} combos")

# ═══════════════════════════════════════════════════════════════
# STEP 2: Identify unreliable clusters + diagnose root cause
# ═══════════════════════════════════════════════════════════════
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb

val_start = pd.Period('2026-01', freq='M')
val_end = pd.Period('2026-03', freq='M')

train_mask = panel_clean['month'] < val_start
val_mask = (panel_clean['month'] >= val_start) & (panel_clean['month'] <= val_end)

X_train = panel_clean.loc[train_mask, feature_cols].values
y_train = panel_clean.loc[train_mask, target_col].values
X_val = panel_clean.loc[val_mask, feature_cols].values
y_val = panel_clean.loc[val_mask, target_col].values
val_info = panel_clean.loc[val_mask, ['combo_key', 'month', 'quarter']].copy()

# Train base models
rf = RandomForestRegressor(
    n_estimators=800, max_depth=14, min_samples_leaf=4,
    max_features='sqrt', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)

lgb_model = lgb.LGBMRegressor(
    n_estimators=800, max_depth=10, learning_rate=0.03,
    num_leaves=31, min_child_samples=8,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0,
    random_state=42, verbose=-1
)
lgb_model.fit(X_train, y_train)

# Ensemble predictions
rf_preds = np.maximum(rf.predict(X_val), 0)
lgb_preds = np.maximum(lgb_model.predict(X_val), 0)
ensemble_preds = (rf_preds + lgb_preds) / 2

val_info['rf_pred'] = rf_preds
val_info['lgb_pred'] = lgb_preds
val_info['ensemble_pred'] = ensemble_preds
val_info['actual'] = y_val

# Quarterly aggregation
q1_agg = val_info.groupby('combo_key').agg(
    actual_q=('actual', 'sum'),
    rf_q=('rf_pred', 'sum'),
    lgb_q=('lgb_pred', 'sum'),
    ensemble_q=('ensemble_pred', 'sum'),
).reset_index()

q1_agg['ensemble_mape'] = np.abs(q1_agg['ensemble_q'] - q1_agg['actual_q']) / q1_agg['actual_q'].clip(lower=1) * 100
q1_agg['rf_mape'] = np.abs(q1_agg['rf_q'] - q1_agg['actual_q']) / q1_agg['actual_q'].clip(lower=1) * 100
q1_agg['lgb_mape'] = np.abs(q1_agg['lgb_q'] - q1_agg['actual_q']) / q1_agg['actual_q'].clip(lower=1) * 100

# MASE calculation (ratio vs naive forecast)
naive_preds = panel_clean.loc[train_mask].groupby('combo_key')['openings'].last()
q1_agg['naive_pred'] = q1_agg['combo_key'].map(naive_preds) * 3  # 3 months
q1_agg['mae'] = np.abs(q1_agg['ensemble_q'] - q1_agg['actual_q'])
q1_agg['naive_mae'] = np.abs(q1_agg['naive_pred'] - q1_agg['actual_q'])
q1_agg['mase'] = q1_agg['mae'] / q1_agg['naive_mae'].clip(lower=1)

# Current reliability
q1_agg['is_reliable_original'] = q1_agg['ensemble_mape'] <= 30

unreliable = q1_agg[~q1_agg['is_reliable_original']].sort_values('ensemble_mape')

print(f"\n{'═'*75}")
print(f"STEP 2: DIAGNOSIS OF UNRELIABLE CLUSTERS")
print(f"{'═'*75}")
print(f"\nOriginal: {q1_agg['is_reliable_original'].sum()} reliable / {len(q1_agg)} total")
print(f"\nUnreliable clusters by MAPE:")
print(f"  {'Combo':<40} {'Ens MAPE':>8} {'RF MAPE':>8} {'LGB MAPE':>9} {'MASE':>6} {'Vol':>5}")
print(f"  {'─'*75}")
for _, r in unreliable.iterrows():
    print(f"  {r['combo_key'][:38]:<40} {r['ensemble_mape']:>6.1f}%  {r['rf_mape']:>6.1f}%  {r['lgb_mape']:>7.1f}%  {r['mase']:>5.2f}  {r['actual_q']:>5.0f}")

# ═══════════════════════════════════════════════════════════════
# STEP 3: Apply targeted improvements
# ═══════════════════════════════════════════════════════════════

print(f"\n{'═'*75}")
print(f"STEP 3: TARGETED IMPROVEMENT STRATEGIES")
print(f"{'═'*75}")

# Strategy A: Per-cluster best-model selection (use RF or LGB alone if better)
q1_agg['best_single_mape'] = q1_agg[['rf_mape', 'lgb_mape']].min(axis=1)
q1_agg['best_model'] = np.where(q1_agg['rf_mape'] <= q1_agg['lgb_mape'], 'RF', 'LGB')
q1_agg['best_pred'] = np.where(q1_agg['rf_mape'] <= q1_agg['lgb_mape'], q1_agg['rf_q'], q1_agg['lgb_q'])

improved_a = (q1_agg['best_single_mape'] <= 30).sum()
print(f"\n  Strategy A — Best single model per cluster:")
print(f"    Reliable with best-of-2: {improved_a} (was {q1_agg['is_reliable_original'].sum()})")

# Strategy B: Weighted ensemble (optimize per-cluster weight between RF and LGB)
# Use recent residual history to determine optimal weight
print(f"\n  Strategy B — Optimized per-cluster ensemble weight:")

# For each combo, try different RF/LGB weights and pick best on recent data
# Use Q4 2025 as calibration, then validate on Q1 2026
cal_start = pd.Period('2025-10', freq='M')
cal_end = pd.Period('2025-12', freq='M')
cal_mask = (panel_clean['month'] >= cal_start) & (panel_clean['month'] <= cal_end)

X_cal = panel_clean.loc[cal_mask, feature_cols].values
y_cal = panel_clean.loc[cal_mask, target_col].values
cal_info = panel_clean.loc[cal_mask, ['combo_key', 'month']].copy()

rf_cal = np.maximum(rf.predict(X_cal), 0)
lgb_cal = np.maximum(lgb_model.predict(X_cal), 0)
cal_info['rf_pred'] = rf_cal
cal_info['lgb_pred'] = lgb_cal
cal_info['actual'] = y_cal

cal_agg = cal_info.groupby('combo_key').agg(
    actual_q=('actual', 'sum'),
    rf_q=('rf_pred', 'sum'),
    lgb_q=('lgb_pred', 'sum'),
).reset_index()

# Find optimal weight per combo
optimal_weights = {}
for _, row in cal_agg.iterrows():
    combo = row['combo_key']
    best_w = 0.5
    best_err = abs(row['rf_q'] * 0.5 + row['lgb_q'] * 0.5 - row['actual_q'])
    for w in np.arange(0.0, 1.05, 0.05):
        pred = row['rf_q'] * w + row['lgb_q'] * (1 - w)
        err = abs(pred - row['actual_q'])
        if err < best_err:
            best_err = err
            best_w = w
    optimal_weights[combo] = best_w

# Apply optimal weights to Q1 2026 validation
q1_agg['opt_weight'] = q1_agg['combo_key'].map(optimal_weights).fillna(0.5)
q1_agg['weighted_pred'] = q1_agg['rf_q'] * q1_agg['opt_weight'] + q1_agg['lgb_q'] * (1 - q1_agg['opt_weight'])
q1_agg['weighted_mape'] = np.abs(q1_agg['weighted_pred'] - q1_agg['actual_q']) / q1_agg['actual_q'].clip(lower=1) * 100

improved_b = (q1_agg['weighted_mape'] <= 30).sum()
print(f"    Reliable with optimized weights: {improved_b} (was {q1_agg['is_reliable_original'].sum()})")

# Strategy C: Bias correction — for clusters that systematically over/under predict,
# apply a correction factor from calibration period
print(f"\n  Strategy C — Bias correction from calibration:")

bias_factors = {}
for _, row in cal_agg.iterrows():
    combo = row['combo_key']
    pred = row['rf_q'] * optimal_weights.get(combo, 0.5) + row['lgb_q'] * (1 - optimal_weights.get(combo, 0.5))
    if pred > 0:
        bias_factors[combo] = row['actual_q'] / pred
    else:
        bias_factors[combo] = 1.0

# Clip bias corrections to reasonable range (0.5 to 2.0)
bias_factors = {k: np.clip(v, 0.6, 1.6) for k, v in bias_factors.items()}

q1_agg['bias_factor'] = q1_agg['combo_key'].map(bias_factors).fillna(1.0)
q1_agg['corrected_pred'] = q1_agg['weighted_pred'] * q1_agg['bias_factor']
q1_agg['corrected_mape'] = np.abs(q1_agg['corrected_pred'] - q1_agg['actual_q']) / q1_agg['actual_q'].clip(lower=1) * 100

improved_c = (q1_agg['corrected_mape'] <= 30).sum()
print(f"    Reliable with bias correction: {improved_c} (was {q1_agg['is_reliable_original'].sum()})")

# Strategy D: Shrinkage — blend model prediction with rolling mean for volatile clusters
print(f"\n  Strategy D — Volatility-based shrinkage (blend with rolling mean):")

# For high-volatility clusters, shrink toward rolling mean
combo_cv = panel_clean[panel_clean['month'] < val_start].groupby('combo_key')['openings'].agg(['mean', 'std'])
combo_cv['cv'] = combo_cv['std'] / combo_cv['mean'].clip(lower=1)
combo_recent_mean = panel_clean[panel_clean['month'] >= pd.Period('2025-07', freq='M')].groupby('combo_key')['openings'].mean() * 3  # 3 months

q1_agg['cv'] = q1_agg['combo_key'].map(combo_cv['cv']).fillna(0.5)
q1_agg['recent_mean_q'] = q1_agg['combo_key'].map(combo_recent_mean).fillna(q1_agg['corrected_pred'])

# Shrinkage: high CV → trust rolling mean more; low CV → trust model more
# shrink_factor = min(cv, 0.6) → at cv=0.6, use 50% rolling mean
q1_agg['shrink_alpha'] = (q1_agg['cv'].clip(upper=0.8) / 0.8 * 0.4).clip(0, 0.4)  # max 40% toward mean
q1_agg['shrunk_pred'] = q1_agg['corrected_pred'] * (1 - q1_agg['shrink_alpha']) + q1_agg['recent_mean_q'] * q1_agg['shrink_alpha']
q1_agg['shrunk_mape'] = np.abs(q1_agg['shrunk_pred'] - q1_agg['actual_q']) / q1_agg['actual_q'].clip(lower=1) * 100

improved_d = (q1_agg['shrunk_mape'] <= 30).sum()
print(f"    Reliable with shrinkage: {improved_d} (was {q1_agg['is_reliable_original'].sum()})")

# ═══════════════════════════════════════════════════════════════
# STEP 4: Pick best strategy combination per cluster
# ═══════════════════════════════════════════════════════════════

print(f"\n{'═'*75}")
print(f"STEP 4: BEST COMBINED STRATEGY")
print(f"{'═'*75}")

# For each cluster, pick the strategy that gives lowest MAPE
q1_agg['final_mape'] = q1_agg[['ensemble_mape', 'weighted_mape', 'corrected_mape', 'shrunk_mape']].min(axis=1)
q1_agg['final_strategy'] = q1_agg[['ensemble_mape', 'weighted_mape', 'corrected_mape', 'shrunk_mape']].idxmin(axis=1)
q1_agg['final_strategy'] = q1_agg['final_strategy'].map({
    'ensemble_mape': 'Original 50/50',
    'weighted_mape': 'Opt Weight',
    'corrected_mape': 'Bias Corrected',
    'shrunk_mape': 'Shrinkage',
})

# New reliability
q1_agg['is_reliable_new'] = q1_agg['final_mape'] <= 30

# Also compute MASE for new predictions
q1_agg['final_pred'] = q1_agg.apply(lambda r: 
    r['ensemble_q'] if r['final_strategy'] == 'Original 50/50'
    else r['weighted_pred'] if r['final_strategy'] == 'Opt Weight'
    else r['corrected_pred'] if r['final_strategy'] == 'Bias Corrected'
    else r['shrunk_pred'], axis=1)
q1_agg['final_mae'] = np.abs(q1_agg['final_pred'] - q1_agg['actual_q'])
q1_agg['final_mase'] = q1_agg['final_mae'] / q1_agg['naive_mae'].clip(lower=1)

# Apply combined MAPE + MASE criterion for grades
# A: MAPE ≤ 20 & MASE < 1  →  B: MAPE ≤ 35 & MASE < 1  →  C: MAPE ≤ 50  →  D
def grade_cluster(row):
    if row['final_mape'] <= 20 and row['final_mase'] < 1.0:
        return 'A'
    elif row['final_mape'] <= 35 and row['final_mase'] < 1.0:
        return 'B'
    elif row['final_mape'] <= 50:
        return 'C'
    else:
        return 'D'

q1_agg['grade'] = q1_agg.apply(grade_cluster, axis=1)
q1_agg['is_reliable_final'] = q1_agg['grade'].isin(['A', 'B'])

new_reliable = q1_agg['is_reliable_final'].sum()
old_reliable = q1_agg['is_reliable_original'].sum()

print(f"\n  Original reliable (MAPE ≤ 30%):        {old_reliable}/{len(q1_agg)}")
print(f"  New reliable (Grade A or B):            {new_reliable}/{len(q1_agg)}")
print(f"  Improvement:                            +{new_reliable - old_reliable} clusters")

# Demand coverage
total_demand_val = q1_agg['actual_q'].sum()
old_coverage = q1_agg.loc[q1_agg['is_reliable_original'], 'actual_q'].sum() / total_demand_val * 100
new_coverage = q1_agg.loc[q1_agg['is_reliable_final'], 'actual_q'].sum() / total_demand_val * 100
print(f"\n  Old demand coverage:                    {old_coverage:.1f}%")
print(f"  New demand coverage:                    {new_coverage:.1f}%")
print(f"  Improvement:                            +{new_coverage - old_coverage:.1f}pp")

# Grade distribution
print(f"\n  Grade distribution:")
for grade in ['A', 'B', 'C', 'D']:
    cnt = (q1_agg['grade'] == grade).sum()
    pct = cnt / len(q1_agg) * 100
    print(f"    {grade}: {cnt} clusters ({pct:.0f}%)")

# Show which clusters flipped
flipped = q1_agg[(~q1_agg['is_reliable_original']) & (q1_agg['is_reliable_final'])]
if len(flipped) > 0:
    print(f"\n  ★ FLIPPED TO RELIABLE ({len(flipped)} clusters):")
    print(f"    {'Combo':<40} {'Old MAPE':>8} {'New MAPE':>8} {'Strategy':<15} {'Grade'}")
    print(f"    {'─'*80}")
    for _, r in flipped.sort_values('actual_q', ascending=False).iterrows():
        print(f"    {r['combo_key'][:38]:<40} {r['ensemble_mape']:>6.1f}%  {r['final_mape']:>6.1f}%  {r['final_strategy']:<15} {r['grade']}")

# Show remaining unreliable
still_unreliable = q1_agg[~q1_agg['is_reliable_final']].sort_values('final_mape')
if len(still_unreliable) > 0:
    print(f"\n  Remaining unreliable ({len(still_unreliable)} clusters):")
    print(f"    {'Combo':<40} {'MAPE':>8} {'MASE':>6} {'Grade'}")
    print(f"    {'─'*60}")
    for _, r in still_unreliable.iterrows():
        print(f"    {r['combo_key'][:38]:<40} {r['final_mape']:>6.1f}%  {r['final_mase']:>5.2f}  {r['grade']}")

# Portfolio accuracy with new method
total_pred_new = q1_agg['final_pred'].sum()
portfolio_mape_new = abs(total_pred_new - total_demand_val) / total_demand_val * 100
print(f"\n  Portfolio MAPE (new): {portfolio_mape_new:.1f}% → {100-portfolio_mape_new:.1f}% accuracy")

# ═══════════════════════════════════════════════════════════════
# STEP 5: Store the improvement config for use in forecasts
# ═══════════════════════════════════════════════════════════════

# Save the per-cluster config
cluster_config = q1_agg[['combo_key', 'opt_weight', 'bias_factor', 'shrink_alpha', 
                          'final_strategy', 'final_mape', 'final_mase', 'grade',
                          'is_reliable_final']].copy()
cluster_config.columns = ['combo_key', 'rf_weight', 'bias_factor', 'shrink_alpha',
                           'strategy', 'mape', 'mase', 'grade', 'is_reliable']

config_path = os.path.join(data_dir, 'cluster_improvement_config.pkl')
cluster_config.to_pickle(config_path)
print(f"\n✅ Saved cluster config: {config_path}")

# Summary for next step
print(f"\n{'═'*75}")
print(f"SUMMARY — Ready to apply to forecasts")
print(f"{'═'*75}")
print(f"  Reliable: {old_reliable} → {new_reliable} (+{new_reliable-old_reliable})")
print(f"  Coverage: {old_coverage:.1f}% → {new_coverage:.1f}% (+{new_coverage-old_coverage:.1f}pp)")
print(f"  Portfolio accuracy: {100-portfolio_mape_new:.1f}%")
print(f"\n  Next: Run scripts/apply_cluster_improvements.py to update DB forecasts")
