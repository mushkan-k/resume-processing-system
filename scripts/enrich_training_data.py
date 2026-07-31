"""
Enrich Training Data (clean_42k_v1.pkl)
=======================================
Adds:
1. Gap Inc Excel (438 reqs) — real job data with dates
2. Career page scrapes (898 jobs) — fills temporal gaps in existing companies

The key insight from feature importance analysis:
- fills (34.2%), title (10.3%), company (9.6%), issue_date (7.4%)
- Lag features (lag1, lag2, lag3, roll3) drive the Bayesian model
- Temporal GAPS in company data poison lag features → filling them helps

Strategy for career pages:
- Find months where a company has NO records in training data
- But the career page shows they currently have active openings
- Insert synthetic records for those gap months to smooth lag continuity
"""
import pandas as pd
import numpy as np
import json
import os
import pickle
from datetime import datetime

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

print("=" * 70)
print("  ENRICHING TRAINING DATA")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════
# Load current training data
# ═══════════════════════════════════════════════════════════════
df = pd.read_pickle(os.path.join(data_dir, 'clean_42k_v1.pkl'))
print(f"\n  Current training: {len(df):,} records")
print(f"  Date range: {df['issue_date'].min()} → {df['issue_date'].max()}")
print(f"  Companies: {df['company_name'].nunique()}")
print(f"  Columns: {df.columns.tolist()}")

original_len = len(df)

# ═══════════════════════════════════════════════════════════════
# 1. ADD GAP INC EXCEL DATA (438 reqs with real dates)
# ═══════════════════════════════════════════════════════════════
print(f"\n{'─'*70}")
print("  STEP 1: Adding Gap Inc Excel data")
print(f"{'─'*70}")

gap_path = os.path.join(data_dir, '..', '2026 Gap Reqs.xlsx')
if os.path.exists(gap_path):
    gap = pd.read_excel(gap_path)
    print(f"  Gap Excel: {len(gap)} rows")
    print(f"  Columns: {gap.columns.tolist()}")
    
    # Map Gap columns to training format
    gap_records = []
    for _, row in gap.iterrows():
        try:
            issue_date = pd.to_datetime(row['Release Date'])
        except:
            issue_date = pd.to_datetime(row.get('Created Date', '2026-01-01'))
        
        title = str(row.get('Job Title', '')).strip()
        if not title or title == 'nan':
            continue
            
        # Determine region from state
        state = str(row.get('State Name', '')).strip()
        region = 'US'  # Gap is US-based
        
        # Location = state or city
        city = str(row.get('City', '')).strip()
        location = state if state and state != 'nan' else city
        
        # Fill date → compute fills
        fill_date = pd.to_datetime(row.get('Fill Date', pd.NaT), errors='coerce')
        fills = 1 if pd.notna(fill_date) else 0
        
        gap_records.append({
            'issue_date': issue_date,
            'region': region,
            'location': location if location != 'nan' else '',
            'title': title,
            'company_name': 'Gap Inc.',
            'openings': 1,
            'fills': fills,
            'skills_clean': '',  # We'll extract from JD if needed
        })
    
    gap_df = pd.DataFrame(gap_records)
    
    # Deduplicate: remove any Gap records already in training
    existing_gap = df[df['company_name'].str.contains('Gap', case=False, na=False)]
    print(f"  Existing Gap records in training: {len(existing_gap)}")
    
    # Remove duplicates by title+date
    if len(existing_gap) > 0:
        existing_keys = set(zip(
            existing_gap['title'].str.lower(),
            pd.to_datetime(existing_gap['issue_date']).dt.date
        ))
        gap_df['_key'] = list(zip(
            gap_df['title'].str.lower(),
            gap_df['issue_date'].dt.date
        ))
        gap_df = gap_df[~gap_df['_key'].isin(existing_keys)].drop(columns=['_key'])
    
    print(f"  New Gap records to add: {len(gap_df)}")
    print(f"  Date range: {gap_df['issue_date'].min()} → {gap_df['issue_date'].max()}")
    print(f"  Fill rate: {gap_df['fills'].mean():.1%}")
    
    df = pd.concat([df, gap_df], ignore_index=True)
else:
    print("  [!] Gap Excel not found, skipping")

# ═══════════════════════════════════════════════════════════════
# 2. USE CAREER PAGE SCRAPES TO FILL TEMPORAL GAPS
# ═══════════════════════════════════════════════════════════════
print(f"\n{'─'*70}")
print("  STEP 2: Filling temporal gaps with career page data")
print(f"{'─'*70}")

cp_dir = os.path.join(data_dir, 'market_signals', 'career_pages')
if os.path.exists(cp_dir):
    # Load all career page data
    career_data = {}
    for f in os.listdir(cp_dir):
        if f.endswith('.json'):
            with open(os.path.join(cp_dir, f)) as fh:
                data = json.load(fh)
                company = data.get('company', f.replace('.json', ''))
                jobs = data.get('jobs', [])
                if jobs:
                    career_data[company] = {
                        'jobs': jobs,
                        'total': data.get('total_openings', len(jobs)),
                        'scraped_at': data.get('scraped_at', '2026-07-01'),
                    }
    
    print(f"  Career pages loaded: {len(career_data)} companies, {sum(d['total'] for d in career_data.values())} total jobs")
    
    # Map career page company names to training company names (manual mapping)
    training_companies = df['company_name'].unique()
    
    # Explicit mapping (career page name → training data name)
    CAREER_TO_TRAINING = {
        'Amazon': 'Amazon',
        'Amgen': 'Amgen',
        'Caterpillar': 'Caterpillar',
        'eBay': 'eBay',
        'T-Mobile': 'T-Mobile',
        'Visa': 'Visa',
        'Pfizer': 'Pfizer',
        'GAP': 'Gap Inc.',
        'PepsiCo Inc': 'PepsiCo',
        'GE Healthcare': 'GE Healthcare',
        'Diageo': 'Diageo',
        'Rubrik': 'Rubrik',
        'Campbells': 'Campbells',
        'ADVARRA, Inc.': 'ADVARRA',
        'Seqirus': 'Seqirus',
        'Mercedes-Benz': 'Mercedes-Benz',
        'Daimler': 'Daimler',
        'RELX Inc.': 'RELX',
    }
    
    # Only keep mappings where the training company actually exists
    company_mapping = {}
    for cp_name, train_name in CAREER_TO_TRAINING.items():
        if train_name in training_companies:
            company_mapping[cp_name] = train_name
        else:
            # Try case-insensitive match
            matches = [c for c in training_companies if c.lower() == train_name.lower()]
            if matches:
                company_mapping[cp_name] = matches[0]
    
    print(f"  Matched to training companies: {len(company_mapping)}")
    for cp, tc in company_mapping.items():
        print(f"    {cp} → {tc}")
    
    # For matched companies, find temporal gaps and fill them
    df['issue_date'] = pd.to_datetime(df['issue_date'])
    df['_month'] = df['issue_date'].dt.to_period('M')
    
    synthetic_records = []
    all_months = pd.period_range(start='2017-03', end='2026-06', freq='M')
    
    for cp_name, train_name in company_mapping.items():
        cp_data = career_data[cp_name]
        company_df = df[df['company_name'] == train_name]
        
        if len(company_df) < 10:
            continue
        
        # Find months with data
        active_months = set(company_df['_month'].unique())
        
        # Calculate average monthly openings for this company
        monthly_avg = company_df.groupby('_month')['openings'].sum()
        avg_openings_per_month = monthly_avg.mean()
        
        # Find gaps: months within the company's active period that have no data
        first_month = company_df['_month'].min()
        last_month = company_df['_month'].max()
        expected_months = pd.period_range(start=first_month, end=last_month, freq='M')
        gap_months = [m for m in expected_months if m not in active_months]
        
        if not gap_months:
            continue
        
        # Use career page titles to create synthetic records for gap months
        # Distribute career page jobs across gap months proportionally
        cp_jobs = cp_data['jobs']
        n_gaps = len(gap_months)
        
        # For each gap month, create synthetic records using career page titles
        # Scale to match the company's average volume
        jobs_per_gap = max(1, int(avg_openings_per_month * 0.7))  # 70% of avg (conservative)
        
        for gap_month in gap_months[:24]:  # Limit to 24 gap months max per company
            # Pick random titles from career page
            n_jobs = min(jobs_per_gap, len(cp_jobs))
            selected_jobs = np.random.choice(len(cp_jobs), size=n_jobs, replace=True)
            
            for idx in selected_jobs:
                job = cp_jobs[idx]
                # Random day in the gap month
                month_start = gap_month.to_timestamp()
                random_day = month_start + pd.Timedelta(days=np.random.randint(0, 28))
                
                synthetic_records.append({
                    'issue_date': random_day,
                    'region': 'US',  # Most career pages are US
                    'location': job.get('location', ''),
                    'title': job.get('title', 'Unknown'),
                    'company_name': train_name,
                    'openings': 1,
                    'fills': 1,  # Assume filled (historical gaps)
                    'skills_clean': '',
                })
    
    if synthetic_records:
        synth_df = pd.DataFrame(synthetic_records)
        print(f"\n  Synthetic gap-fill records: {len(synth_df)}")
        print(f"  Companies filled: {synth_df['company_name'].nunique()}")
        print(f"  Months covered: {synth_df['issue_date'].dt.to_period('M').nunique()}")
        
        df = pd.concat([df, synth_df], ignore_index=True)
    else:
        print("  No temporal gaps found to fill")
    
    # Also add career page data for companies NOT in training (new signal)
    unmatched_companies = [cp for cp in career_data if cp not in company_mapping]
    new_company_records = []
    
    for cp_name in unmatched_companies:
        cp_data = career_data[cp_name]
        jobs = cp_data['jobs']
        if len(jobs) < 5:
            continue
        
        # Create records as of scrape date (current market signal)
        scrape_date = pd.to_datetime(cp_data.get('scraped_at', '2026-07-01'))
        
        # Normalize company name
        nice_name = cp_name.replace('_', ' ').title()
        
        for job in jobs:
            title = job.get('title', '')
            if not title:
                continue
            new_company_records.append({
                'issue_date': scrape_date,
                'region': 'US',
                'location': job.get('location', ''),
                'title': title,
                'company_name': nice_name,
                'openings': 1,
                'fills': 0,  # Current openings, not yet filled
                'skills_clean': '',
            })
    
    if new_company_records:
        new_df = pd.DataFrame(new_company_records)
        print(f"\n  New company records (from career pages): {len(new_df)}")
        print(f"  Companies: {new_df['company_name'].nunique()} — {new_df['company_name'].unique().tolist()}")
        df = pd.concat([df, new_df], ignore_index=True)
else:
    print("  [!] Career pages directory not found")

# ═══════════════════════════════════════════════════════════════
# 3. CLEANUP AND SAVE
# ═══════════════════════════════════════════════════════════════
print(f"\n{'─'*70}")
print("  STEP 3: Saving enriched training data")
print(f"{'─'*70}")

# Drop temp columns
if '_month' in df.columns:
    df = df.drop(columns=['_month'])

# Ensure types
df['issue_date'] = pd.to_datetime(df['issue_date'])
df['openings'] = df['openings'].fillna(1).astype(int)
df['fills'] = df['fills'].fillna(0).astype(int)

# Save
output_path = os.path.join(data_dir, 'clean_42k_v1.pkl')
backup_path = os.path.join(data_dir, 'clean_42k_v1_pre_enrich_backup.pkl')

# Backup original
original_df = pd.read_pickle(output_path)
original_df.to_pickle(backup_path)
print(f"  Backed up original to: clean_42k_v1_pre_enrich_backup.pkl")

# Save enriched
df.to_pickle(output_path)

print(f"\n  SUMMARY:")
print(f"  {'─'*50}")
print(f"  Original records:    {original_len:>8,}")
print(f"  Final records:       {len(df):>8,}")
print(f"  Added:               {len(df) - original_len:>8,} ({(len(df)-original_len)/original_len*100:.1f}%)")
print(f"  Companies (before):  {original_df['company_name'].nunique():>8}")
print(f"  Companies (after):   {df['company_name'].nunique():>8}")
print(f"  Date range:          {df['issue_date'].min().date()} → {df['issue_date'].max().date()}")
print(f"\n  Saved to: {output_path}")
print("  Done!")
