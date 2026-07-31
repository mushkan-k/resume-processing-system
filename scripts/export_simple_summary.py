"""Simple Excel: Total JDs + Top Skills per company (last 12 months)."""
import pandas as pd
import re

df = pd.read_pickle('data/clean_42k_v1.pkl')
df['issue_date'] = pd.to_datetime(df['issue_date'])
cutoff = pd.Timestamp('2025-06-01')
recent = df[df['issue_date'] >= cutoff].copy()

def parse_skills(s):
    if not s or pd.isna(s):
        return []
    s = str(s)
    s = re.sub(r'\s*over\s+\d+\s*year\(s\)\s*', '', s, flags=re.IGNORECASE)
    groups = re.split(r'\)\s*AND\s*\(|\)\s*AND\(', s)
    skills = []
    for g in groups:
        g = g.strip('() ')
        for p in re.split(r'\s+OR\s+', g, flags=re.IGNORECASE):
            sk = p.strip('() ').strip()
            if sk and 1 < len(sk) < 60:
                skip = ['YEARS','YEAR','SENIOR','LEAD','JUNIOR','BACHELOR','MASTER',
                        'DEGREE','STRONG','EXCELLENT','GOOD','EXPERIENCE','SKILLS']
                if not any(w in sk.upper() for w in skip):
                    skills.append(sk.title())
    return list(set(skills))

recent['parsed_skills'] = recent['skills_clean'].apply(parse_skills)

companies = {
    'Caterpillar': ['Caterpillar', 'IN Caterpillar'],
    'T-Mobile': ['T-Mobile'],
    'Wabtec': ['Wabtec', 'Wabtec Corporation'],
}

# --- Sheet 1: Overview ---
overview_rows = []
for name, aliases in companies.items():
    cdf = recent[recent['company_name'].isin(aliases)]
    overview_rows.append({
        'Company': name,
        'Total JDs': len(cdf),
        'Total Openings': int(cdf['openings'].sum()),
        'JDs with Skills': cdf['parsed_skills'].apply(len).gt(0).sum(),
    })
overview_df = pd.DataFrame(overview_rows)

# --- Sheet 2: Top Skills per Company ---
skills_rows = []
for name, aliases in companies.items():
    cdf = recent[recent['company_name'].isin(aliases)]
    all_skills = []
    for sl in cdf['parsed_skills']:
        all_skills.extend(sl)
    top = pd.Series(all_skills).value_counts().head(20)
    for rank, (skill, count) in enumerate(top.items(), 1):
        skills_rows.append({
            'Company': name,
            'Rank': rank,
            'Skill': skill,
            'JD Count': count,
        })
skills_df = pd.DataFrame(skills_rows)

# --- Write Excel ---
path = 'data/company_jd_summary.xlsx'
with pd.ExcelWriter(path, engine='openpyxl') as writer:
    overview_df.to_excel(writer, sheet_name='Overview', index=False)
    skills_df.to_excel(writer, sheet_name='Top Skills', index=False)

# Auto-fit columns
from openpyxl import load_workbook
wb = load_workbook(path)
for ws in wb.worksheets:
    for col in ws.columns:
        max_len = max(min(len(str(cell.value or '')), 40) for cell in col)
        ws.column_dimensions[col[0].column_letter].width = max_len + 3
wb.save(path)

print(f"Done: {path}")
