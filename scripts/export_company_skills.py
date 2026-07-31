"""
Export Company-Specific JD Skills to Excel
==========================================
Creates an Excel file with:
- Sheet 1: Summary (top skills per company per cluster)
- Sheet 2: Caterpillar JDs with skills
- Sheet 3: T-Mobile JDs with skills
- Sheet 4: Wabtec JDs with skills
- Sheet 5: All Companies raw data
"""
import pandas as pd
import numpy as np
import re
import os

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

# Load data
df = pd.read_pickle(os.path.join(data_dir, 'clean_42k_v1.pkl'))
df['issue_date'] = pd.to_datetime(df['issue_date'])
df['month'] = df['issue_date'].dt.to_period('M')

# Load cluster mapping
mapping = pd.read_pickle(os.path.join(data_dir, 'title_to_cluster.pkl'))
title_to_role = dict(zip(mapping['raw_title'], mapping['role_cluster']))
df['role_cluster'] = df['title'].map(title_to_role)

# ═══════════════════════════════════════════════════════════════
# Parse skills from the boolean JD requirement strings
# ═══════════════════════════════════════════════════════════════
def parse_skills_from_jd(skill_string):
    """
    Extract individual skills from boolean JD requirement strings like:
    '(JAVA OR SPRING BOOT) AND(AWS OR AZURE) AND(CI/CD OR JENKINS)'
    → ['Java', 'Spring Boot', 'AWS', 'Azure', 'CI/CD', 'Jenkins']
    """
    if not skill_string or pd.isna(skill_string):
        return []
    
    s = str(skill_string)
    
    # Remove experience requirements like "over 3 year(s)"
    s = re.sub(r'\s*over\s+\d+\s*year\(s\)\s*', '', s, flags=re.IGNORECASE)
    
    # Remove AND/OR operators and parentheses, split by common delimiters
    # First, split by AND to get requirement groups
    groups = re.split(r'\)\s*AND\s*\(|\)\s*AND\(', s)
    
    skills = []
    for group in groups:
        # Clean up parentheses
        group = group.strip('() ')
        # Split by OR
        parts = re.split(r'\s+OR\s+', group, flags=re.IGNORECASE)
        for part in parts:
            skill = part.strip('() ').strip()
            # Filter out noise
            if skill and len(skill) > 1 and len(skill) < 60:
                # Skip generic phrases
                skip_words = ['YEARS', 'YEAR', 'SENIOR', 'LEAD', 'JUNIOR', 'ENTRY LEVEL',
                              'BACHELOR', 'MASTER', 'DEGREE', 'B.E', 'BTECH', 'STRONG',
                              'EXCELLENT', 'GOOD', 'EXPERIENCE', 'SKILLS']
                if not any(sw in skill.upper() for sw in skip_words):
                    # Title case for readability
                    skills.append(skill.title().strip())
    
    return list(set(skills))  # deduplicate


# ═══════════════════════════════════════════════════════════════
# Filter to target companies, last 12 months
# ═══════════════════════════════════════════════════════════════
target_companies = {
    'Caterpillar': ['Caterpillar', 'IN Caterpillar'],
    'T-Mobile': ['T-Mobile'],
    'Wabtec': ['Wabtec', 'Wabtec Corporation'],
}

# Last 12 months
cutoff = pd.Timestamp('2025-06-01')
recent = df[df['issue_date'] >= cutoff].copy()

print("=" * 70)
print("EXPORTING COMPANY JD SKILLS TO EXCEL")
print("=" * 70)
print(f"\n  Total records (last 12m): {len(recent)}")
print(f"  With skills: {recent['skills_clean'].notna().sum()}")

# Parse skills for all rows
recent['parsed_skills'] = recent['skills_clean'].apply(parse_skills_from_jd)
recent['skill_count'] = recent['parsed_skills'].apply(len)

# ═══════════════════════════════════════════════════════════════
# Build per-company sheets
# ═══════════════════════════════════════════════════════════════
all_company_data = []
company_summaries = []

for company_label, company_names in target_companies.items():
    mask = recent['company_name'].isin(company_names)
    company_df = recent[mask].copy()
    
    print(f"\n  {company_label}: {len(company_df)} jobs, "
          f"{company_df['skill_count'].gt(0).sum()} with skills")
    
    # Build export dataframe
    export_rows = []
    for _, row in company_df.iterrows():
        skills = row['parsed_skills']
        export_rows.append({
            'Company': company_label,
            'Title': row['title'],
            'Role Cluster': row['role_cluster'] if pd.notna(row['role_cluster']) else 'Unmapped',
            'Region': row['region'],
            'Location': row['location'] if pd.notna(row.get('location')) else '',
            'Issue Date': row['issue_date'].strftime('%Y-%m-%d'),
            'Month': str(row['month']),
            'Openings': row['openings'],
            'Fills': row.get('fills', 0),
            'Skills (Parsed)': ', '.join(skills) if skills else '',
            'Skill Count': len(skills),
            'Raw JD Requirements': str(row['skills_clean']) if pd.notna(row['skills_clean']) else '',
        })
    
    company_export = pd.DataFrame(export_rows)
    company_export = company_export.sort_values(['Role Cluster', 'Issue Date'], ascending=[True, False])
    all_company_data.append((company_label, company_export))
    
    # Build skill summary for this company
    all_skills = []
    for _, row in company_df.iterrows():
        cluster = row['role_cluster'] if pd.notna(row['role_cluster']) else 'Unmapped'
        for skill in row['parsed_skills']:
            all_skills.append({
                'Company': company_label,
                'Role Cluster': cluster,
                'Skill': skill,
            })
    
    if all_skills:
        skills_df = pd.DataFrame(all_skills)
        # Count per company + cluster
        skill_counts = skills_df.groupby(['Company', 'Role Cluster', 'Skill']).size().reset_index(name='Frequency')
        skill_counts = skill_counts.sort_values(['Role Cluster', 'Frequency'], ascending=[True, False])
        company_summaries.append(skill_counts)

# ═══════════════════════════════════════════════════════════════
# Build Summary sheet: Top 10 skills per company per cluster
# ═══════════════════════════════════════════════════════════════
summary_rows = []
for company_label, company_names in target_companies.items():
    mask = recent['company_name'].isin(company_names)
    company_df = recent[mask]
    
    for cluster in sorted(company_df['role_cluster'].dropna().unique()):
        cluster_df = company_df[company_df['role_cluster'] == cluster]
        all_skills = []
        for skills_list in cluster_df['parsed_skills']:
            all_skills.extend(skills_list)
        
        if all_skills:
            counts = pd.Series(all_skills).value_counts()
            for skill, freq in counts.head(10).items():
                summary_rows.append({
                    'Company': company_label,
                    'Role Cluster': cluster,
                    'Skill': skill,
                    'Frequency': freq,
                    'Total JDs in Cluster': len(cluster_df),
                    '% of JDs': round(freq / max(len(cluster_df), 1) * 100, 0),
                })

summary_df = pd.DataFrame(summary_rows)

# ═══════════════════════════════════════════════════════════════
# Write to Excel
# ═══════════════════════════════════════════════════════════════
excel_path = os.path.join(output_dir, 'company_jd_skills_analysis.xlsx')

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    # Sheet 1: Summary
    summary_df.to_excel(writer, sheet_name='Skills Summary', index=False)
    
    # Sheet per company
    for company_label, company_export in all_company_data:
        sheet_name = company_label[:31]  # Excel max 31 chars
        company_export.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # Combined skill frequency
    if company_summaries:
        all_skills_combined = pd.concat(company_summaries, ignore_index=True)
        all_skills_combined.to_excel(writer, sheet_name='All Skills Frequency', index=False)

print(f"\n{'='*70}")
print(f"✅ EXPORTED: {excel_path}")
print(f"{'='*70}")
print(f"\n  Sheets:")
print(f"    1. Skills Summary — Top 10 skills per company per cluster")
print(f"    2. Caterpillar — All JDs with parsed skills")
print(f"    3. T-Mobile — All JDs with parsed skills")
print(f"    4. Wabtec — All JDs with parsed skills")
print(f"    5. All Skills Frequency — Complete skill counts")

# Quick preview
print(f"\n{'─'*70}")
print("PREVIEW: Top skills per company")
print("─" * 70)

for company in ['Caterpillar', 'T-Mobile', 'Wabtec']:
    comp_summary = summary_df[summary_df['Company'] == company]
    if len(comp_summary) > 0:
        # Top skills across all clusters
        top = comp_summary.groupby('Skill')['Frequency'].sum().sort_values(ascending=False).head(8)
        print(f"\n  {company} (top skills across all roles):")
        for skill, freq in top.items():
            print(f"    • {skill}: {freq} mentions")
