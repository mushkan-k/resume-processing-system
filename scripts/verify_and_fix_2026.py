"""
Verify & Fix 2026 Data — Compare pickle vs JobDiva API for every month
Then update the pickle with complete data.
"""
import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

BASE_URL = "https://api.jobdiva.com/apiv2"
PICKLE_PATH = ROOT / "data" / "clean_42k_v1.pkl"


def authenticate():
    resp = requests.get(f"{BASE_URL}/authenticate", params={
        "clientid": os.getenv("JOBDIVA_CLIENT_ID"),
        "username": os.getenv("JOBDIVA_USERNAME"),
        "password": os.getenv("JOBDIVA_PASSWORD")
    }, timeout=30)
    resp.raise_for_status()
    return resp.text.strip().strip('"')


def fetch_month(token, year, month):
    """Fetch all issued jobs for a given month in 15-day chunks."""
    headers = {"Authorization": f"Bearer {token}"}
    import calendar
    last_day = calendar.monthrange(year, month)[1]
    
    all_jobs = []
    chunks = [
        (f"{month:02d}/01/{year}", f"{month:02d}/15/{year}"),
        (f"{month:02d}/16/{year}", f"{month:02d}/{last_day}/{year}"),
    ]
    for fd, td in chunks:
        resp = requests.get(f"{BASE_URL}/bi/IssuedJobsList",
                           params={"fromDate": fd, "toDate": td},
                           headers=headers, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            recs = data.get("data", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
            all_jobs.extend(recs)
    
    # Deduplicate by JOBID
    seen = set()
    unique = []
    for j in all_jobs:
        jid = j.get("JOBID", id(j))
        if jid not in seen:
            seen.add(jid)
            unique.append(j)
    return unique


def api_jobs_to_rows(jobs):
    """Convert API job records to rows matching the pickle schema."""
    rows = []
    for j in jobs:
        issue_date = j.get("ISSUEDATE", "")
        if issue_date:
            try:
                # Handle various date formats
                for fmt in ["%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%m/%d/%Y"]:
                    try:
                        dt = pd.to_datetime(issue_date, format=fmt)
                        break
                    except:
                        continue
                else:
                    dt = pd.to_datetime(issue_date)
            except:
                continue
        else:
            continue
        
        rows.append({
            "issue_date": dt,
            "region": j.get("COUNTRY", ""),
            "location": f"{j.get('CITY', '')}, {j.get('STATE', '')}".strip(", "),
            "title": j.get("TITLE", ""),
            "company_name": j.get("COMPANYNAME", ""),
            "openings": int(j.get("OPENINGS", 1) or 1),
            "fills": 0,
            "skills_clean": "",
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 70)
    print("  VERIFY & FIX 2026 DATA — Pickle vs JobDiva API")
    print("=" * 70)
    
    # Load current pickle
    df = pd.read_pickle(PICKLE_PATH)
    df["month"] = pd.to_datetime(df["issue_date"]).dt.to_period("M")
    
    print(f"\n  Current pickle: {len(df):,} rows")
    print(f"  Date range: {df['month'].min()} to {df['month'].max()}")
    
    # Authenticate
    print(f"\n  Authenticating with JobDiva...")
    token = authenticate()
    print(f"  ✓ OK\n")
    
    # Check every 2026 month
    months_to_check = []
    for m in range(1, 8):  # Jan-Jul 2026
        period = pd.Period(f"2026-{m:02d}", freq="M")
        if period <= pd.Period(datetime.now().strftime("%Y-%m"), freq="M"):
            months_to_check.append((2026, m))
    
    print(f"  {'Month':<10} {'Pickle Rows':>12} {'Pickle Open':>12} {'API Jobs':>10} {'API Open':>10} {'Status':>10}")
    print(f"  {'-'*68}")
    
    months_to_fix = []
    api_data_cache = {}
    
    for year, month in months_to_check:
        period = pd.Period(f"{year}-{month:02d}", freq="M")
        pkl_subset = df[df["month"] == period]
        pkl_rows = len(pkl_subset)
        pkl_openings = int(pkl_subset["openings"].sum())
        
        # Fetch from API
        api_jobs = fetch_month(token, year, month)
        api_count = len(api_jobs)
        api_openings = sum(int(j.get("OPENINGS", 1) or 1) for j in api_jobs)
        api_data_cache[(year, month)] = api_jobs
        
        # Determine status
        if pkl_rows == 0 and api_count == 0:
            status = "NO DATA"
        elif api_count == 0:
            status = "API EMPTY"
        elif pkl_openings >= api_openings * 0.9:
            status = "✅ OK"
        elif pkl_openings < api_openings * 0.5:
            status = "⚠️ INCOMPLETE"
            months_to_fix.append((year, month, period))
        else:
            status = "⚠️ LOW"
            months_to_fix.append((year, month, period))
        
        print(f"  {str(period):<10} {pkl_rows:>12,} {pkl_openings:>12,} {api_count:>10,} {api_openings:>10,} {status:>10}")
    
    # Fix incomplete months
    if not months_to_fix:
        print(f"\n  ✅ All months look good! No fixes needed.")
        return
    
    print(f"\n  {'='*70}")
    print(f"  FIXING {len(months_to_fix)} INCOMPLETE MONTH(S)")
    print(f"  {'='*70}")
    
    # Backup
    backup_path = PICKLE_PATH.with_suffix(".pkl.bak")
    df_original = pd.read_pickle(PICKLE_PATH)
    df_original.to_pickle(backup_path)
    print(f"\n  Backed up original to: {backup_path}")
    
    df_fixed = df.drop(columns=["month"]).copy()
    
    for year, month, period in months_to_fix:
        api_jobs = api_data_cache[(year, month)]
        new_rows = api_jobs_to_rows(api_jobs)
        
        if len(new_rows) == 0:
            print(f"  {period}: No valid rows from API, skipping")
            continue
        
        # Remove old incomplete data for this month
        old_mask = pd.to_datetime(df_fixed["issue_date"]).dt.to_period("M") == period
        old_count = old_mask.sum()
        df_fixed = df_fixed[~old_mask]
        
        # Add new complete data
        df_fixed = pd.concat([df_fixed, new_rows], ignore_index=True)
        
        print(f"  {period}: Replaced {old_count} rows with {len(new_rows)} rows "
              f"({new_rows['openings'].sum():,} openings)")
    
    # Sort and save
    df_fixed = df_fixed.sort_values("issue_date").reset_index(drop=True)
    df_fixed.to_pickle(PICKLE_PATH)
    
    print(f"\n  Updated pickle saved: {len(df_fixed):,} rows")
    print(f"  (was: {len(df_original):,} rows)")
    
    # Verify
    print(f"\n  {'='*70}")
    print(f"  VERIFICATION — Updated Pickle")
    print(f"  {'='*70}")
    df_verify = pd.read_pickle(PICKLE_PATH)
    df_verify["month"] = pd.to_datetime(df_verify["issue_date"]).dt.to_period("M")
    mt = df_verify[df_verify["month"] >= pd.Period("2026-01")].groupby("month").agg(
        rows=("openings", "count"), total=("openings", "sum"))
    print(f"\n  {'Month':<10} {'Rows':>8} {'Openings':>10}")
    print(f"  {'-'*30}")
    for m, r in mt.iterrows():
        print(f"  {str(m):<10} {int(r['rows']):>8} {int(r['total']):>10,}")
    
    print(f"\n  ✅ Done! Now re-run: python scripts/generate_hier_forecasts.py --months 3")


if __name__ == "__main__":
    main()
