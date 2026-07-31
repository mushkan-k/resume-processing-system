import pandas as pd
import mysql.connector
from collections import Counter
import re
import random

conn = mysql.connector.connect(host='localhost', port=3305, database='resume_processing', user='resume_user', password='resume_password')
df = pd.read_sql("SELECT skills, title, experience_level, position_type FROM updated_job_records", conn)
conn.close()

skills = df['skills']
filled = skills.notna() & (skills.str.strip() != '')
print(f"Total records: {len(df)}")
print(f"Skills filled: {filled.sum()} ({filled.sum()/len(df)*100:.1f}%)")
print(f"Skills empty/null: {(~filled).sum()} ({(~filled).sum()/len(df)*100:.1f}%)")
print(f"\nAvg length (filled): {skills[filled].str.len().mean():.0f} chars")
print(f"Median length: {skills[filled].str.len().median():.0f}")
print(f"Max length: {skills[filled].str.len().max()}")

placeholder = skills[filled].str.contains('SKILLS TO BE ASSIGNED', case=False, na=False)
has_boolean = skills[filled].str.contains(r'\bOR\b|\bAND\b', na=False, regex=True)
has_years = skills[filled].str.contains(r'year', case=False, na=False)

print(f"\n{'='*80}")
print("CONTENT BREAKDOWN")
print(f"{'='*80}")
print(f"Placeholder only: {placeholder.sum()} ({placeholder.sum()/filled.sum()*100:.1f}%)")
print(f"Has boolean (AND/OR): {has_boolean.sum()} ({has_boolean.sum()/filled.sum()*100:.1f}%)")
print(f"Mentions experience/years: {has_years.sum()} ({has_years.sum()/filled.sum()*100:.1f}%)")

# What's in the non-boolean, non-placeholder ones?
simple = filled & ~placeholder & ~has_boolean
print(f"\nSimple text (no boolean, no placeholder): {simple.sum()}")

print(f"\n{'='*80}")
print("15 RANDOM FULL SAMPLES")
print(f"{'='*80}")
random.seed(42)
indices = random.sample(list(skills[filled].index), 15)
for i, idx in enumerate(indices, 1):
    title = df.loc[idx, 'title']
    exp = df.loc[idx, 'experience_level']
    pos = df.loc[idx, 'position_type']
    s = skills[idx]
    print(f"\n--- Sample {i} | {title[:50]} | exp={exp} | type={pos} ---")
    print(s[:500])

# Check experience_level and position_type columns
print(f"\n{'='*80}")
print("EXPERIENCE_LEVEL COLUMN")
print(f"{'='*80}")
print(df['experience_level'].value_counts(dropna=False).head(20))

print(f"\n{'='*80}")
print("POSITION_TYPE COLUMN")
print(f"{'='*80}")
print(df['position_type'].value_counts(dropna=False).head(20))

# What kind of info can we actually extract from skills?
print(f"\n{'='*80}")
print("WHAT'S EXTRACTABLE FROM SKILLS COLUMN")
print(f"{'='*80}")

# Check for common tech terms
tech_patterns = {
    'Python': r'\bpython\b',
    'Java': r'\bjava\b',
    'SQL': r'\bsql\b',
    'AWS': r'\baws\b',
    'Azure': r'\bazure\b',
    'JavaScript': r'\bjavascript|\.js\b',
    'React': r'\breact\b',
    '.NET': r'\.net|dotnet',
    'C#': r'\bc#\b',
    'Kubernetes': r'\bkubernetes|k8s\b',
    'Docker': r'\bdocker\b',
    'SAP': r'\bsap\b',
    'Salesforce': r'\bsalesforce\b',
    'ServiceNow': r'\bservicenow\b',
    'Agile/Scrum': r'\bagile|scrum\b',
    'Machine Learning': r'\bmachine learning|ml\b',
    'Data Science': r'\bdata scien',
    'Power BI': r'\bpower bi|powerbi\b',
    'Tableau': r'\btableau\b',
    'Excel': r'\bexcel\b',
}

print(f"\nTech term frequency in skills column (out of {filled.sum()} filled records):")
for term, pattern in sorted(tech_patterns.items(), key=lambda x: -skills[filled].str.contains(x[1], case=False, na=False, regex=True).sum()):
    count = skills[filled].str.contains(pattern, case=False, na=False, regex=True).sum()
    print(f"  {term:20s} | {count:5d} ({count/filled.sum()*100:.1f}%)")

# Experience years extraction
print(f"\n{'='*80}")
print("EXPERIENCE REQUIREMENTS IN SKILLS COLUMN")
print(f"{'='*80}")
year_pattern = r'(\d+)\s*(?:\+\s*)?year'
years_found = skills[filled].str.extractall(year_pattern)
if not years_found.empty:
    years_vals = years_found[0].astype(int)
    print(f"Records mentioning years: {has_years.sum()}")
    print(f"Year values found: {len(years_vals)}")
    print(f"Distribution of experience years mentioned:")
    print(years_vals.value_counts().sort_index().head(15))
