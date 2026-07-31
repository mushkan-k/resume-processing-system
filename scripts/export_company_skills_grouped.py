"""
Export Company JD Skills — GROUPED by Location & Role
=====================================================
Creates a cleaner Excel with:
- Sheet 1: Role × Location Summary (one row per group, top skills in one cell)
- Sheet 2: Location-based view (what roles/skills are hot per city)
- Sheet 3: Role-based view (skills & locations per role)
- Sheet 4: Company Hiring Heatmap (role × month matrix)
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
df['month'] = df['issue_date'].dt.to_period('M').astype(str)

# Load cluster mapping
mapping = pd.read_pickle(os.path.join(data_dir, 'title_to_cluster.pkl'))
title_to_role = dict(zip(mapping['raw_title'], mapping['role_cluster']))
df['role_cluster'] = df['title'].map(title_to_role)


def parse_skills_from_jd(skill_string):
    """Extract skills from boolean JD requirement strings."""
    if not skill_string or pd.isna(skill_string):
        return []
    s = str(skill_string)
    s = re.sub(r'\s*over\s+\d+\s*year\(s\)\s*', '', s, flags=re.IGNORECASE)
    groups = re.split(r'\)\s*AND\s*\(|\)\s*AND\(', s)
    skills = []
    for group in groups:
        group = group.strip('() ')
        parts = re.split(r'\s+OR\s+', group, flags=re.IGNORECASE)
        for part in parts:
            skill = part.strip('() ').strip()
            if skill and len(skill) > 1 and len(skill) < 60:
                skip_words = ['YEARS', 'YEAR', 'SENIOR', 'LEAD', 'JUNIOR', 'ENTRY LEVEL',
                              'BACHELOR', 'MASTER', 'DEGREE', 'B.E', 'BTECH', 'STRONG',
                              'EXCELLENT', 'GOOD', 'EXPERIENCE', 'SKILLS']
                if not any(sw in skill.upper() for sw in skip_words):
                    skills.append(skill.title().strip())
    return list(set(skills))


# ═══════════════════════════════════════════════════════════════
# Filter to target companies, last 12 months
# ═══════════════════════════════════════════════════════════════
target_companies = {
    'Caterpillar': ['Caterpillar', 'IN Caterpillar'],
    'T-Mobile': ['T-Mobile'],
    'Wabtec': ['Wabtec', 'Wabtec Corporation'],
}

cutoff = pd.Timestamp('2025-06-01')
recent = df[df['issue_date'] >= cutoff].copy()
recent['parsed_skills'] = recent['skills_clean'].apply(parse_skills_from_jd)
recent['skill_count'] = recent['parsed_skills'].apply(len)

# Tag company
def get_company_label(name):
    for label, names in target_companies.items():
        if name in names:
            return label
    return None

recent['company_label'] = recent['company_name'].apply(get_company_label)
company_df = recent[recent['company_label'].notna()].copy()

# Use region (country) and location (city) — fill blanks
company_df['country'] = company_df['region'].fillna('Unknown')
company_df['city'] = company_df['location'].fillna('Unknown')
company_df['role'] = company_df['role_cluster'].fillna('Unmapped')

print("=" * 70)
print("EXPORTING GROUPED COMPANY JD SKILLS")
print("=" * 70)
print(f"\n  Total company JDs (last 12m): {len(company_df)}")
print(f"  With skills parsed: {company_df['skill_count'].gt(0).sum()}")

# ═══════════════════════════════════════════════════════════════
# SHEET 1: Role × Location Summary
# One row per Company + Role + Location — top skills consolidated
# ═══════════════════════════════════════════════════════════════
print("\n  Building Sheet 1: Role × Location Summary...")

grouped_rows = []
for (company, role, country, city), grp in company_df.groupby(
    ['company_label', 'role', 'country', 'city'], sort=True
):
    total_jds = len(grp)
    total_openings = int(grp['openings'].sum())
    
    # Collect all skills and count
    all_skills = []
    for skills_list in grp['parsed_skills']:
        all_skills.extend(skills_list)
    
    skill_counts = pd.Series(all_skills).value_counts()
    top5 = skill_counts.head(5)
    top5_str = ', '.join([f"{s} ({c})" for s, c in top5.items()]) if len(top5) > 0 else '—'
    all_skills_str = ', '.join(skill_counts.head(15).index.tolist()) if len(skill_counts) > 0 else '—'
    
    # Date range
    date_min = grp['issue_date'].min().strftime('%Y-%m')
    date_max = grp['issue_date'].max().strftime('%Y-%m')
    
    grouped_rows.append({
        'Company': company,
        'Role': role,
        'Country': country,
        'City': city,
        'JDs': total_jds,
        'Openings': total_openings,
        'Period': f"{date_min} → {date_max}",
        'Top 5 Skills (with count)': top5_str,
        'All Skills (up to 15)': all_skills_str,
        'Unique Skills': len(skill_counts),
    })

sheet1 = pd.DataFrame(grouped_rows)
sheet1 = sheet1.sort_values(['Company', 'Role', 'Openings'], ascending=[True, True, False])

print(f"    → {len(sheet1)} grouped rows (vs thousands of individual skill rows)")

# ═══════════════════════════════════════════════════════════════
# SHEET 2: Location View — what's hot in each city
# One row per Company + Country + City
# ═══════════════════════════════════════════════════════════════
print("  Building Sheet 2: Location View...")

location_rows = []
for (company, country, city), grp in company_df.groupby(
    ['company_label', 'country', 'city'], sort=True
):
    total_jds = len(grp)
    total_openings = int(grp['openings'].sum())
    
    # Top roles
    role_counts = grp['role'].value_counts()
    top_roles = ', '.join([f"{r} ({c})" for r, c in role_counts.head(5).items()])
    
    # Top skills across all roles
    all_skills = []
    for skills_list in grp['parsed_skills']:
        all_skills.extend(skills_list)
    skill_counts = pd.Series(all_skills).value_counts()
    top_skills = ', '.join([f"{s} ({c})" for s, c in skill_counts.head(8).items()]) if len(skill_counts) > 0 else '—'
    
    location_rows.append({
        'Company': company,
        'Country': country,
        'City': city,
        'Total JDs': total_jds,
        'Total Openings': total_openings,
        'Top Roles': top_roles,
        'Top Skills (with count)': top_skills,
        'Distinct Roles': grp['role'].nunique(),
    })

sheet2 = pd.DataFrame(location_rows)
sheet2 = sheet2.sort_values(['Company', 'Total Openings'], ascending=[True, False])

print(f"    → {len(sheet2)} location groups")

# ═══════════════════════════════════════════════════════════════
# SHEET 3: Role View — skills & locations per role
# One row per Company + Role
# ═══════════════════════════════════════════════════════════════
print("  Building Sheet 3: Role View...")

role_rows = []
for (company, role), grp in company_df.groupby(['company_label', 'role'], sort=True):
    total_jds = len(grp)
    total_openings = int(grp['openings'].sum())
    
    # Top locations
    loc_counts = grp.groupby(['country', 'city']).size().sort_values(ascending=False)
    top_locs = ', '.join([f"{c[1]} ({c[0]}): {v}" for c, v in loc_counts.head(5).items()])
    
    # Top skills
    all_skills = []
    for skills_list in grp['parsed_skills']:
        all_skills.extend(skills_list)
    skill_counts = pd.Series(all_skills).value_counts()
    top_skills = ', '.join([f"{s} ({c})" for s, c in skill_counts.head(10).items()]) if len(skill_counts) > 0 else '—'
    
    # Sample titles
    sample_titles = grp['title'].value_counts().head(3).index.tolist()
    
    role_rows.append({
        'Company': company,
        'Role': role,
        'Total JDs': total_jds,
        'Total Openings': total_openings,
        'Top Locations': top_locs,
        'Top 10 Skills': top_skills,
        'Sample Titles': ' | '.join(sample_titles),
        'Distinct Locations': loc_counts.shape[0],
    })

sheet3 = pd.DataFrame(role_rows)
sheet3 = sheet3.sort_values(['Company', 'Total Openings'], ascending=[True, False])

print(f"    → {len(sheet3)} role groups")

# ═══════════════════════════════════════════════════════════════
# SHEET 4: Monthly Hiring Heatmap (Role × Month)
# ═══════════════════════════════════════════════════════════════
print("  Building Sheet 4: Monthly Heatmap...")

heatmap_rows = []
for company in ['Caterpillar', 'T-Mobile', 'Wabtec']:
    cdf = company_df[company_df['company_label'] == company]
    pivot = cdf.groupby(['role', 'month'])['openings'].sum().unstack(fill_value=0)
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    
    for role in pivot.index:
        row_data = {'Company': company, 'Role': role, 'Total': int(pivot.loc[role].sum())}
        for month in sorted(pivot.columns):
            row_data[month] = int(pivot.loc[role].get(month, 0))
        heatmap_rows.append(row_data)

sheet4 = pd.DataFrame(heatmap_rows)

print(f"    → {len(sheet4)} heatmap rows")

# ═══════════════════════════════════════════════════════════════
# Write to Excel with formatting
# ═══════════════════════════════════════════════════════════════
excel_path = os.path.join(output_dir, 'company_skills_grouped.xlsx')
print(f"\n  Writing to: {excel_path}")

with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    sheet1.to_excel(writer, sheet_name='Role × Location', index=False)
    sheet2.to_excel(writer, sheet_name='By Location', index=False)
    sheet3.to_excel(writer, sheet_name='By Role', index=False)
    sheet4.to_excel(writer, sheet_name='Monthly Heatmap', index=False)

# Auto-fit column widths
from openpyxl import load_workbook
wb = load_workbook(excel_path)
for ws in wb.worksheets:
    for col in ws.columns:
        max_len = 0
        col_letter = col[0].column_letter
        for cell in col:
            if cell.value:
                max_len = max(max_len, min(len(str(cell.value)), 50))
        ws.column_dimensions[col_letter].width = max_len + 2
wb.save(excel_path)

print(f"\n{'=' * 70}")
print(f"  ✅ DONE — {excel_path}")
print(f"{'=' * 70}")
print(f"""
  Sheet 1: 'Role × Location' — {len(sheet1)} rows
           One row per Company + Role + City. Top 5 skills in one cell.
           
  Sheet 2: 'By Location' — {len(sheet2)} rows  
           One row per City. Shows top roles + top skills for that location.
           
  Sheet 3: 'By Role' — {len(sheet3)} rows
           One row per Role. Shows top locations + top 10 skills.
           
  Sheet 4: 'Monthly Heatmap' — {len(sheet4)} rows
           Openings per role per month. Spot trends at a glance.
""")
