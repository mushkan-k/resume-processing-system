"""
Augment Training Data (clean_42k_v2.pkl)
=========================================
Adds:
1. Gap Inc Excel (438 reqs) — new company with full JD data
2. Career page scrapes — used to fill time-series gaps for existing combos
   (if a company is hiring for a role on their career page NOW but has gaps
    in our monthly history, we impute openings=1 for those gap months)

This gives the model more continuous lag features → better predictions.
"""
import pandas as pd
import numpy as np
import os, json, re
from datetime import datetime

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')


def extract_skills_from_jd(text):
    """Extract tech skills from JD text."""
    tech_keywords = [
        'python', 'java', 'sql', 'aws', 'azure', 'react', 'angular', 'node',
        'kubernetes', 'docker', 'terraform', 'ci/cd', 'agile', 'scrum',
        'machine learning', 'data science', 'tableau', 'power bi', 'excel',
        'salesforce', 'sap', 'oracle', 'jira', 'git', 'linux', 'api',
        'microservices', 'devops', 'cloud', 'security', 'networking',
    ]
    text_lower = text.lower()
    found = [kw for kw in tech_keywords if kw in text_lower]
    return ', '.join(found[:10])


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Load existing training data
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70)
print("AUGMENTING TRAINING DATA")
print("=" * 70)

df = pd.read_pickle(os.path.join(data_dir, 'clean_42k_v1.pkl'))
print(f"\n  Original clean_42k: {len(df):,} records")
print(f"  Columns: {df.columns.tolist()}")
print(f"  Date range: {df['issue_date'].min()} to {df['issue_date'].max()}")
print(f"  Companies: {df['company_name'].nunique()}")

# Load cluster mapping
ttc = pd.read_pickle(os.path.join(data_dir, 'title_to_cluster.pkl'))
title_to_cluster = dict(zip(ttc['raw_title'], ttc['role_cluster']))

# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Add Gap Inc Excel data
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("STEP 2: Adding Gap Inc data (438 reqs)")
print("─" * 70)

gap_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '2026 Gap Reqs.xlsx')
if os.path.exists(gap_path):
    gap = pd.read_excel(gap_path)
    print(f"  Loaded: {len(gap)} Gap reqs")
    
    # Map columns to match clean_42k format
    gap_records = []
    for _, row in gap.iterrows():
        title = str(row.get('Job Title', '')).strip()
        if not title or title == 'nan':
            continue
        
        # Parse dates
        issue_date = pd.to_datetime(row.get('Release Date', row.get('Created Date')), errors='coerce')
        if pd.isna(issue_date):
            continue
        
        # Determine region from state
        state = str(row.get('State Name', '')).strip()
        city = str(row.get('City', '')).strip()
        region = 'US'  # Gap is US-based
        location = state if state and state != 'nan' else ''
        
        # Skills from JD text
        jd_text = str(row.get('Job Description', ''))
        skills = extract_skills_from_jd(jd_text) if len(jd_text) > 50 else ''
        
        gap_records.append({
            'issue_date': issue_date,
            'region': region,
            'location': location,
            'title': title,
            'company_name': 'Gap Inc',
            'openings': 1,
            'fills': 1 if pd.notna(row.get('Fill Date')) else 0,
            'skills_clean': skills,
        })
    
    gap_df = pd.DataFrame(gap_records)
    # Deduplicate against existing
    existing_gap = df[df['company_name'].str.contains('Gap', case=False, na=False)]
    print(f"  Existing Gap records in pickle: {len(existing_gap)}")
    
    # Only add records not already present (by title + date)
    if len(existing_gap) > 0:
        existing_keys = set(zip(existing_gap['title'], existing_gap['issue_date'].dt.date))
        new_mask = ~gap_df.apply(lambda r: (r['title'], r['issue_date'].date()) in existing_keys, axis=1)
        gap_df = gap_df[new_mask]
    
    print(f"  New Gap records to add: {len(gap_df)}")
    df = pd.concat([df, gap_df], ignore_index=True)
else:
    print("  [!] Gap Excel not found, skipping")

# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Use career scrapes to fill time-series gaps
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("STEP 3: Filling time-series gaps with career page evidence")
print("─" * 70)

cp_dir = os.path.join(data_dir, 'market_signals', 'career_pages')
if os.path.exists(cp_dir):
    # Load career page data
    career_evidence = {}  # company -> list of titles
    for f in os.listdir(cp_dir):
        if not f.endswith('.json'):
            continue
        data = json.load(open(os.path.join(cp_dir, f), encoding='utf-8'))
        company = data.get('company', f.replace('.json', ''))
        jobs = data.get('jobs', [])
        titles = [j['title'] for j in jobs if j.get('title')]
        if titles:
            career_evidence[company] = titles
    
    print(f"  Career page companies: {len(career_evidence)}")
    
    # Map career page company names to our company names
    our_companies = df['company_name'].unique()
    company_map = {}
    for cp_company in career_evidence:
        cp_lower = cp_company.lower().replace('_', ' ').replace('-', ' ')
        for our_c in our_companies:
            our_lower = our_c.lower()
            # Fuzzy match
            if (cp_lower in our_lower or our_lower in cp_lower or
                cp_lower.split()[0] in our_lower):
                company_map[cp_company] = our_c
                break
    
    print(f"  Matched to our companies: {len(company_map)}")
    
    # For each matched company, find which clusters they're hiring for NOW
    # Then check for gaps in their monthly time-series
    df['_month'] = pd.to_datetime(df['issue_date']).dt.to_period('M')
    df['_cluster'] = df['title'].map(title_to_cluster)
    
    gap_fills = []
    all_months = pd.period_range(df['_month'].min(), df['_month'].max(), freq='M')
    
    for cp_company, our_company in company_map.items():
        # Get clusters this company is hiring for (from career page)
        cp_titles = career_evidence[cp_company]
        cp_clusters = set()
        for t in cp_titles:
            cluster = title_to_cluster.get(t)
            if cluster:
                cp_clusters.add(cluster)
            else:
                # Try partial match
                t_lower = t.lower()
                for raw_t, clust in title_to_cluster.items():
                    if raw_t.lower() in t_lower or t_lower in raw_t.lower():
                        cp_clusters.add(clust)
                        break
        
        if not cp_clusters:
            continue
        
        # For each cluster this company hires for, check monthly continuity
        company_data = df[df['company_name'] == our_company]
        for cluster in cp_clusters:
            cluster_data = company_data[company_data['_cluster'] == cluster]
            if len(cluster_data) < 3:
                continue  # Need some history to have gaps worth filling
            
            active_months = set(cluster_data['_month'].unique())
            first_month = cluster_data['_month'].min()
            last_month = cluster_data['_month'].max()
            
            # Find gap months (between first and last active month)
            expected_months = pd.period_range(first_month, last_month, freq='M')
            gap_months = [m for m in expected_months if m not in active_months]
            
            if not gap_months:
                continue
            
            # Only fill gaps that are <=3 months (short gaps, not long absences)
            # Group consecutive gaps
            consecutive_gaps = []
            current_run = [gap_months[0]]
            for i in range(1, len(gap_months)):
                if gap_months[i] == gap_months[i-1] + 1:
                    current_run.append(gap_months[i])
                else:
                    consecutive_gaps.append(current_run)
                    current_run = [gap_months[i]]
            consecutive_gaps.append(current_run)
            
            # Only fill short gaps (<=3 months)
            avg_openings = cluster_data['openings'].median()
            sample_row = cluster_data.iloc[0]
            
            for run in consecutive_gaps:
                if len(run) > 3:
                    continue  # Skip long gaps — might be genuine absence
                
                for gap_month in run:
                    gap_fills.append({
                        'issue_date': gap_month.to_timestamp() + pd.Timedelta(days=15),
                        'region': sample_row['region'],
                        'location': sample_row['location'],
                        'title': cluster_data['title'].mode().iloc[0] if len(cluster_data['title'].mode()) > 0 else sample_row['title'],
                        'company_name': our_company,
                        'openings': max(1, int(avg_openings * 0.5)),  # Conservative: half of median
                        'fills': max(1, int(avg_openings * 0.3)),
                        'skills_clean': sample_row.get('skills_clean', ''),
                    })
    
    if gap_fills:
        gap_fills_df = pd.DataFrame(gap_fills)
        print(f"  Gap months filled: {len(gap_fills_df)} synthetic records")
        print(f"  Companies with fills: {gap_fills_df['company_name'].nunique()}")
        print(f"  Avg openings imputed: {gap_fills_df['openings'].mean():.1f}")
        df = pd.concat([df, gap_fills_df], ignore_index=True)
    else:
        print("  No gap fills needed")
else:
    print("  [!] Career pages directory not found")

# ═══════════════════════════════════════════════════════════════════════
# STEP 4: Clean up and save
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("STEP 4: Final cleanup and save")
print("─" * 70)

# Drop temp columns
df = df.drop(columns=['_month', '_cluster'], errors='ignore')

# Sort by date
df = df.sort_values('issue_date').reset_index(drop=True)

print(f"\n  Final dataset: {len(df):,} records")
print(f"  Companies: {df['company_name'].nunique()}")
print(f"  Date range: {df['issue_date'].min()} to {df['issue_date'].max()}")
print(f"  Added: {len(df) - 42835:,} new records")

# Save
output_path = os.path.join(data_dir, 'clean_42k_v1.pkl')
df.to_pickle(output_path)
print(f"\n  [OK] Saved to {output_path}")
print(f"  (Backup: clean_42k_v1_backup.pkl already exists from recluster)")
