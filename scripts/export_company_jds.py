"""
Export JDs for Caterpillar, T-Mobile, Wabtec — clean grouped view.
==================================================================
One row per Role Cluster per Country. Cities clubbed together.
Columns: Role, Country, Cities, JDs, Skills, Titles
"""
import pandas as pd
import re

df = pd.read_pickle('data/clean_42k_v1.pkl')
df['issue_date'] = pd.to_datetime(df['issue_date'])

mapping = pd.read_pickle('data/title_to_cluster.pkl')
df['role_cluster'] = df['title'].map(dict(zip(mapping['raw_title'], mapping['role_cluster'])))

cutoff = pd.Timestamp('2025-06-01')
recent = df[df['issue_date'] >= cutoff].copy()


def parse_skills(s):
    if not s or pd.isna(s):
        return []
    s = re.sub(r'\s*over\s+\d+\s*year\(s\)\s*', '', str(s), flags=re.IGNORECASE)
    groups = re.split(r'\)\s*AND\s*\(|\)\s*AND\(', s)
    skills = []
    for g in groups:
        g = g.strip('() ')
        for p in re.split(r'\s+OR\s+', g, flags=re.IGNORECASE):
            sk = p.strip('() ').strip()
            if sk and 1 < len(sk) < 60:
                skip = ['YEARS', 'YEAR', 'SENIOR', 'LEAD', 'JUNIOR', 'BACHELOR', 'MASTER',
                        'DEGREE', 'STRONG', 'EXCELLENT', 'GOOD', 'EXPERIENCE', 'SKILLS']
                if not any(w in sk.upper() for w in skip):
                    skills.append(sk.title())
    return list(set(skills))


recent['parsed_skills'] = recent['skills_clean'].apply(parse_skills)
recent['location'] = recent['location'].fillna('Unknown')
recent['role_cluster'] = recent['role_cluster'].fillna('Unmapped')

companies = {
    'Caterpillar': ['Caterpillar', 'IN Caterpillar'],
    'T-Mobile': ['T-Mobile'],
    'Wabtec': ['Wabtec', 'Wabtec Corporation'],
}

print("=" * 70)
print("  EXPORTING CLEAN GROUPED JDs")
print("=" * 70)

sheets = {}

for label, aliases in companies.items():
    cdf = recent[recent['company_name'].isin(aliases)].copy()
    print(f"\n  {label}: {len(cdf)} JDs")

    rows = []
    # Group by Role + Country (cities clubbed together)
    grouped = cdf.groupby(['role_cluster', 'region'], sort=True)

    for (role, country), grp in grouped:
        jd_count = len(grp)

        # Club cities together with counts
        city_counts = grp['location'].value_counts()
        cities_str = ', '.join([f"{city} ({c})" for city, c in city_counts.items()])

        # Club titles together
        title_counts = grp['title'].value_counts()
        titles_str = ', '.join([f"{t} ({c})" for t, c in title_counts.items()])

        # All skills with counts
        all_skills = []
        for sl in grp['parsed_skills']:
            all_skills.extend(sl)
        skill_counts = pd.Series(all_skills).value_counts()
        skills_str = ', '.join([f"{s} ({c})" for s, c in skill_counts.items()]) if len(skill_counts) > 0 else '-'

        rows.append({
            'Role': role,
            'Country': country,
            'Cities': cities_str,
            'JDs': jd_count,
            'Skills': skills_str,
            'Titles': titles_str,
        })

    sheet_df = pd.DataFrame(rows)
    sheet_df = sheet_df.sort_values(['Country', 'Role'], ascending=[True, True])
    sheets[label] = sheet_df
    print(f"    -> {len(sheet_df)} rows")

# Write Excel
path = 'data/company_jds_detailed.xlsx'
print(f"\n  Writing to: {path}")

with pd.ExcelWriter(path, engine='openpyxl') as writer:
    for label, sheet_df in sheets.items():
        sheet_df.to_excel(writer, sheet_name=label, index=False)

# Auto-fit columns
from openpyxl import load_workbook
wb = load_workbook(path)
for ws in wb.worksheets:
    for col in ws.columns:
        max_len = max(min(len(str(cell.value or '')), 60) for cell in col)
        ws.column_dimensions[col[0].column_letter].width = max_len + 3
wb.save(path)

print(f"\n  DONE - {path}")
for label, sheet_df in sheets.items():
    print(f"  '{label}' - {len(sheet_df)} rows (one per Role + Country)")
