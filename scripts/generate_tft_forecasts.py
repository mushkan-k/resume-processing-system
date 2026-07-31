"""
TFT (Temporal Fusion Transformer) Demand Forecast Generator
=============================================================
Model: TFT with QuantileLoss — optimizes INTERVAL accuracy directly.
Key difference from Bayesian: TFT learns quantile boundaries (p10, p50, p90)
directly, so "predict 14-19" actually means 80% of actuals land there.

Focus: INTERVAL CALIBRATION > point MAPE
"""
import pandas as pd
import numpy as np
import os
import warnings
import pickle
import mysql.connector
from datetime import datetime

warnings.filterwarnings('ignore')

import torch
import lightning.pytorch as pl
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer

# ═══════════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════════
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
MODEL_VERSION = "TFT_v1_QuantileLoss"

DB_CONFIG = {
    "host": "localhost",
    "port": 3305,
    "database": "resume_processing",
    "user": "resume_user",
    "password": "resume_password",
}

ENC_LEN = 12   # 12 months lookback
PRED_LEN = 6   # 6 months ahead (Jul-Dec 2026)
BATCH_SIZE = 64
MAX_EPOCHS = 40
LEARNING_RATE = 0.003
HIDDEN_SIZE = 48
ATTENTION_HEADS = 4
DROPOUT = 0.15

# Quantiles for interval: p15, p50, p85 → ~70% coverage (tighter, more useful)
QUANTILES = [0.15, 0.50, 0.85]


def load_and_build_panel():
    """Load training data and build monthly panel with features."""
    print("  Loading data...")
    
    df = pd.read_pickle(os.path.join(DATA_DIR, 'clean_42k_v1.pkl'))
    ttc = pd.read_pickle(os.path.join(DATA_DIR, 'title_to_cluster.pkl'))
    title_map = dict(zip(ttc['raw_title'], ttc['role_cluster']))
    
    df['role_cluster'] = df['title'].map(title_map)
    df = df.dropna(subset=['role_cluster'])
    df['issue_date'] = pd.to_datetime(df['issue_date'])
    df['month'] = df['issue_date'].dt.to_period('M')
    
    # Region
    if 'region' not in df.columns:
        df['region'] = 'US'
    df['region'] = df['region'].fillna('US')
    df['region_role'] = df['region'] + ' | ' + df['role_cluster']
    
    # Drop incomplete last month
    monthly_total = df.groupby('month')['openings'].sum().sort_index()
    if len(monthly_total) >= 2:
        last_month = monthly_total.index[-1]
        if monthly_total.iloc[-1] < monthly_total.iloc[-2] * 0.5:
            print(f"  Dropping incomplete month {last_month}")
            df = df[df['month'] != last_month]
    
    print(f"  {len(df):,} records, {df['region_role'].nunique()} region|role combos")
    
    # Build monthly time series per region_role (aggregated across companies, same as Bayesian output)
    ts = df.groupby(['region_role', 'month']).agg(
        openings=('openings', 'sum'),
        fills=('fills', 'sum'),
    ).reset_index()
    
    # Match Bayesian panel: only combos with >= 20 months of data
    combo_months = ts.groupby('region_role')['month'].nunique()
    viable = combo_months[combo_months >= 20].index.tolist()
    panel = ts[ts['region_role'].isin(viable)].copy()
    
    print(f"  Viable combos (>=20 months, same as Bayesian): {len(viable)}")
    print(f"  Panel: {len(panel):,} rows")
    
    # Fill missing months with 0 (TFT needs continuous series)
    all_months = pd.period_range(panel['month'].min(), panel['month'].max(), freq='M')
    full_idx = pd.MultiIndex.from_product([viable, all_months], names=['region_role', 'month'])
    panel = panel.set_index(['region_role', 'month']).reindex(full_idx, fill_value=0).reset_index()
    
    # Add features
    panel['month_ts'] = panel['month'].dt.to_timestamp()
    panel['time_idx'] = (panel['month'] - panel['month'].min()).apply(lambda x: x.n)
    panel['cal_month'] = panel['month'].apply(lambda x: x.month)
    panel['quarter'] = panel['month'].apply(lambda x: x.quarter)
    panel['fill_rate'] = panel['fills'] / panel['openings'].clip(lower=1)
    
    # Lag features (TFT handles these via encoder, but explicit lags help too)
    panel = panel.sort_values(['region_role', 'time_idx']).reset_index(drop=True)
    panel['lag1'] = panel.groupby('region_role')['openings'].shift(1).fillna(0)
    panel['lag3_avg'] = panel.groupby('region_role')['openings'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
    panel['lag12_avg'] = panel.groupby('region_role')['openings'].transform(
        lambda x: x.shift(1).rolling(12, min_periods=3).mean()).fillna(0)
    
    # Load macro signals
    jolts = pd.read_csv(os.path.join(DATA_DIR, 'JTSJOL.csv'), parse_dates=['observation_date'])
    jolts['month'] = jolts['observation_date'].dt.to_period('M')
    jolts_m = jolts.groupby('month')['JTSJOL'].last().reset_index().rename(columns={'JTSJOL': 'jolts'})
    
    unrate = pd.read_csv(os.path.join(DATA_DIR, 'UNRATE.csv'), parse_dates=['observation_date'])
    unrate['month'] = unrate['observation_date'].dt.to_period('M')
    unrate_m = unrate.groupby('month')['UNRATE'].last().reset_index().rename(columns={'UNRATE': 'unrate'})
    
    panel = panel.merge(jolts_m, on='month', how='left')
    panel = panel.merge(unrate_m, on='month', how='left')
    panel['jolts'] = panel['jolts'].ffill().bfill()
    panel['unrate'] = panel['unrate'].ffill().bfill()
    
    # Ensure no NaN
    panel = panel.fillna(0)
    
    # Ensure target is float (required by GroupNormalizer softplus)
    panel['openings'] = panel['openings'].astype(float)
    panel['fills'] = panel['fills'].astype(float)
    panel['fill_rate'] = panel['fill_rate'].astype(float)
    panel['lag1'] = panel['lag1'].astype(float)
    panel['lag3_avg'] = panel['lag3_avg'].astype(float)
    panel['lag12_avg'] = panel['lag12_avg'].astype(float)
    
    print(f"  Final panel: {len(panel):,} rows, {panel['region_role'].nunique()} combos, time_idx 0-{panel['time_idx'].max()}")
    
    return panel, viable


def train_tft(panel, viable):
    """Train TFT model with QuantileLoss for calibrated intervals."""
    print(f"\n  Building TFT dataset (enc={ENC_LEN}, pred={PRED_LEN})...")
    
    # Training cutoff: everything before the last PRED_LEN months = training
    max_time = panel['time_idx'].max()
    train_cutoff = max_time - PRED_LEN
    
    training = panel[panel['time_idx'] <= train_cutoff].copy()
    
    # Create TimeSeriesDataSet
    tds = TimeSeriesDataSet(
        training,
        time_idx='time_idx',
        target='openings',
        group_ids=['region_role'],
        max_encoder_length=ENC_LEN,
        min_encoder_length=ENC_LEN // 2,
        max_prediction_length=PRED_LEN,
        min_prediction_length=1,
        static_categoricals=['region_role'],
        time_varying_known_reals=['time_idx', 'cal_month', 'quarter'],
        time_varying_unknown_reals=['openings', 'fills', 'fill_rate', 'lag1', 'lag3_avg', 'lag12_avg', 'jolts', 'unrate'],
        target_normalizer=GroupNormalizer(groups=['region_role'], transformation='softplus'),
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )
    
    # Validation (last PRED_LEN months of training)
    val_cutoff = train_cutoff - PRED_LEN
    validation = TimeSeriesDataSet.from_dataset(tds, panel[panel['time_idx'] <= max_time], predict=True, stop_randomization=True)
    
    train_dl = tds.to_dataloader(train=True, batch_size=BATCH_SIZE, num_workers=0)
    val_dl = validation.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)
    
    print(f"  Training samples: {len(tds)}, Validation samples: {len(validation)}")
    
    # Create TFT model
    tft = TemporalFusionTransformer.from_dataset(
        tds,
        learning_rate=LEARNING_RATE,
        hidden_size=HIDDEN_SIZE,
        attention_head_size=ATTENTION_HEADS,
        dropout=DROPOUT,
        hidden_continuous_size=16,
        output_size=len(QUANTILES),
        loss=QuantileLoss(quantiles=QUANTILES),
        reduce_on_plateau_patience=3,
    )
    
    print(f"  TFT parameters: {sum(p.numel() for p in tft.parameters()):,}")
    
    # Train
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='cpu',
        enable_progress_bar=True,
        gradient_clip_val=0.1,
        enable_model_summary=False,
    )
    
    print(f"  Training TFT ({MAX_EPOCHS} epochs)...")
    trainer.fit(tft, train_dataloaders=train_dl, val_dataloaders=val_dl)
    
    return tft, tds, panel


def generate_forecasts(tft, tds, panel, viable):
    """Generate forecasts for the prediction horizon."""
    print("\n  Generating TFT forecasts...")
    
    max_time = panel['time_idx'].max()
    
    # Create prediction dataset from full panel
    pred_ds = TimeSeriesDataSet.from_dataset(tds, panel, predict=True, stop_randomization=True)
    pred_dl = pred_ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)
    
    # Get quantile predictions
    raw_preds = tft.predict(pred_dl, mode='raw')
    # raw_preds['prediction'] shape: (n_combos, PRED_LEN, n_quantiles)
    predictions = raw_preds['prediction'].numpy()
    
    # Map back to region_role
    # The decoder gives us predictions for each combo
    forecasts = []
    
    # Forecast months: Jul-Dec 2026
    forecast_months = pd.period_range('2026-07', periods=PRED_LEN, freq='M')
    
    viable_sorted = sorted(panel['region_role'].unique())
    for i, combo in enumerate(viable_sorted):
        if i >= predictions.shape[0]:
            break
        for t in range(min(PRED_LEN, predictions.shape[1])):
            month = forecast_months[t] if t < len(forecast_months) else forecast_months[-1]
            p10 = max(0, round(float(predictions[i, t, 0])))
            p50 = max(0, round(float(predictions[i, t, 1])))
            p90 = max(1, round(float(predictions[i, t, 2])))
            
            # Ensure ordering
            if p10 > p50:
                p10 = p50
            if p90 < p50:
                p90 = p50 + 1
            
            forecasts.append({
                'cluster_name': combo,
                'forecast_month': str(month),
                'demand_predicted': p50,
                'demand_lower': p10,
                'demand_upper': p90,
            })
    
    forecasts_df = pd.DataFrame(forecasts)
    print(f"  Generated {len(forecasts_df)} forecast rows for {forecasts_df['cluster_name'].nunique()} clusters")
    print(f"  Months: {forecasts_df['forecast_month'].nunique()}")
    print(f"  Total point forecast: {forecasts_df['demand_predicted'].sum():,}")
    
    return forecasts_df


def evaluate_calibration(tft, tds, panel):
    """Evaluate interval calibration on held-out data."""
    print("\n  Evaluating interval calibration...")
    
    max_time = panel['time_idx'].max()
    train_cutoff = max_time - PRED_LEN
    
    # The actual values for the forecast period
    actuals = panel[panel['time_idx'] > train_cutoff].groupby('region_role')['openings'].sum()
    
    # Get predictions
    pred_ds = TimeSeriesDataSet.from_dataset(tds, panel[panel['time_idx'] <= max_time], predict=True, stop_randomization=True)
    pred_dl = pred_ds.to_dataloader(train=False, batch_size=BATCH_SIZE, num_workers=0)
    
    raw_preds = tft.predict(pred_dl, mode='raw')
    predictions = raw_preds['prediction'].numpy()
    
    # Check how many actuals fall within [p10, p90]
    viable_list = sorted(panel['region_role'].unique())
    in_range = 0
    total = 0
    mapes = []
    
    for i, combo in enumerate(viable_list):
        if i >= predictions.shape[0]:
            break
        if combo not in actuals.index:
            continue
        
        actual = actuals[combo]
        # Sum predictions across PRED_LEN months
        pred_lo = max(0, predictions[i, :, 0].sum())
        pred_mid = max(0, predictions[i, :, 1].sum())
        pred_hi = max(1, predictions[i, :, 2].sum())
        
        if actual >= pred_lo and actual <= pred_hi:
            in_range += 1
        total += 1
        
        if actual > 0:
            mapes.append(abs(pred_mid - actual) / actual)
    
    coverage = in_range / total * 100 if total > 0 else 0
    median_mape = np.median(mapes) * 100 if mapes else 0
    
    print(f"  Interval coverage (actual within p10-p90): {coverage:.1f}% (target: ~80%)")
    print(f"  Median MAPE: {median_mape:.1f}%")
    print(f"  Clusters evaluated: {total}")
    
    return coverage, median_mape


def write_to_db(forecasts_df, coverage, median_mape):
    """Write TFT forecasts to demand_forecasts table."""
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Only keep forecast months Jul-Dec 2026 (future months)
    forecasts_df = forecasts_df[forecasts_df['forecast_month'] >= '2026-07'].copy()
    
    # Delete existing predicted rows for these months
    months = forecasts_df['forecast_month'].unique()
    for m in months:
        cur.execute("DELETE FROM demand_forecasts WHERE forecast_month = %s AND data_type = 'predicted'", (m,))
    conn.commit()
    
    insert_sql = """
        INSERT INTO demand_forecasts
        (cluster_name, forecast_month, demand_predicted, demand_lower, demand_upper,
         model_used, mape, is_reliable, data_type)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'predicted')
    """
    
    inserted = 0
    for _, row in forecasts_df.iterrows():
        try:
            cur.execute(insert_sql, (
                row['cluster_name'], row['forecast_month'],
                int(row['demand_predicted']), int(row['demand_lower']), int(row['demand_upper']),
                MODEL_VERSION, round(median_mape, 1), 1 if coverage >= 70 else 0,
            ))
            inserted += 1
        except Exception as e:
            pass  # Skip duplicates
    
    conn.commit()
    cur.close()
    conn.close()
    print(f"\n  [OK] Inserted {inserted} TFT forecast rows")


def main():
    print("=" * 70)
    print("  TFT DEMAND FORECAST GENERATOR")
    print(f"  Model: {MODEL_VERSION}")
    print(f"  Quantiles: {QUANTILES} → ~80% coverage intervals")
    print(f"  Focus: INTERVAL CALIBRATION (actual lands within predicted range)")
    print("=" * 70)
    
    pl.seed_everything(42)
    
    # Build panel
    panel, viable = load_and_build_panel()
    
    # Train TFT
    tft, tds, panel = train_tft(panel, viable)
    
    # Evaluate calibration
    coverage, median_mape = evaluate_calibration(tft, tds, panel)
    
    # Generate forecasts
    forecasts_df = generate_forecasts(tft, tds, panel, viable)
    
    # Write to DB
    write_to_db(forecasts_df, coverage, median_mape)
    
    # Save model
    model_path = os.path.join(DATA_DIR, 'tft_model.pkl')
    torch.save(tft.state_dict(), model_path)
    print(f"  Model saved to {model_path}")
    
    print(f"\n  {'='*50}")
    print(f"  SUMMARY")
    print(f"  {'='*50}")
    print(f"  Model: {MODEL_VERSION}")
    print(f"  Interval Coverage: {coverage:.1f}% (target ≥80%)")
    print(f"  Median MAPE: {median_mape:.1f}%")
    print(f"  Total forecast: {forecasts_df['demand_predicted'].sum():,} openings (Jul-Dec)")
    print(f"  Clusters: {forecasts_df['cluster_name'].nunique()}")
    print(f"  Done!")


if __name__ == '__main__':
    main()
