"""
Check data sufficiency per company for the forecasting model.
Model needs enough monthly data points to predict future quarters.
Rule of thumb: need at least 6+ months of history for a reasonable forecast.
"""
import mysql.connector
import pandas as pd
from datetime import datetime

DB_CONFIG = {
    "host": "localhost", "port": 3305, "database": "resume_processing",
    "user": "resume_user", "password": "resume_password",
}

conn = mysql.connector.connect(**DB_CONFIG)

# Get monthly data depth per company
df = pd.read_sql("""
    SELECT company_name,
           COUNT(*) as total_records,
           SUM(openings) as total_openings,
           COUNT(DISTINCT DATE_FORMAT(issue_date, '%%Y-%%m')) as months_with_data,
           MIN(issue_date) as first_date,
           MAX(issue_date) as last_date,
           COUNT(DISTINCT title) as unique_titles,
           DATEDIFF(MAX(issue_date), MIN(issue_date)) as date_span_days
    FROM updated_job_records
    WHERE company_name IN (
        'Caterpillar','Amgen','Samsung','IN Caterpillar','John Deere',
        'IN Standard Chartered','IN Amazon','Facebook','Verizon One','Microsoft',
        'IN MasterCard','IN PayPal','Wabtec Corporation','T-Mobile','PayPal',
        'IN Diageo PLC','IN Microsoft','Salesforce','eBay','Pfizer',
        'Amazon','Visa','Blue Shield of California','IN Linkedin','RELX Inc.',
        'IN Salesforce','IN Intuit','Best Buy','Oracle','GAP',
        'Catalent Corporation','Novo Nordisk','Zebra Technologies','Excellus',
        'State Street','ADVARRA, Inc.','Dell','Expedia','Rubrik',
        'Kroger','SAIC','Campbells','IHG','IN Rubrik',
        'PepsiCo Inc','IN Q2 Financials','Pearson','Diageo','Seqirus',
        'Solidigm','Daimler','GE Aerospace','Integra LifeSciences',
        'Mercedes-Benz','Red Hat','CIN7','IN PepsiCo','NAIC',
        'Blue Origin','GE Healthcare'
    )
    GROUP BY company_name
    ORDER BY total_openings DESC
""", conn)

# Also get monthly breakdown for each company
monthly = pd.read_sql("""
    SELECT company_name, DATE_FORMAT(issue_date, '%%Y-%%m') as month, 
           SUM(openings) as openings, COUNT(*) as records
    FROM updated_job_records
    WHERE company_name IN (
        'Caterpillar','Amgen','Samsung','IN Caterpillar','John Deere',
        'IN Standard Chartered','IN Amazon','Facebook','Verizon One','Microsoft',
        'IN MasterCard','IN PayPal','Wabtec Corporation','T-Mobile','PayPal',
        'IN Diageo PLC','IN Microsoft','Salesforce','eBay','Pfizer',
        'Amazon','Visa','Blue Shield of California','IN Linkedin','RELX Inc.',
        'IN Salesforce','IN Intuit','Best Buy','Oracle','GAP',
        'Catalent Corporation','Novo Nordisk','Zebra Technologies','Excellus',
        'State Street','ADVARRA, Inc.','Dell','Expedia','Rubrik',
        'Kroger','SAIC','Campbells','IHG','IN Rubrik',
        'PepsiCo Inc','IN Q2 Financials','Pearson','Diageo','Seqirus',
        'Solidigm','Daimler','GE Aerospace','Integra LifeSciences',
        'Mercedes-Benz','Red Hat','CIN7','IN PepsiCo','NAIC',
        'Blue Origin','GE Healthcare'
    )
    AND issue_date >= '2025-01-01'
    GROUP BY company_name, DATE_FORMAT(issue_date, '%%Y-%%m')
    ORDER BY company_name, month
""", conn)
conn.close()

print("=" * 100)
print("  DATA SUFFICIENCY CHECK — 61 KEEP COMPANIES")
print("=" * 100)
print(f"\n  Model requirement: Need 6+ months of data for reliable forecasting")
print(f"  Ideal: 12+ months with consistent monthly volume\n")

# Classify sufficiency
SUFFICIENT = 6  # months minimum
IDEAL = 12

sufficient = []
marginal = []
insufficient = []
not_found = []

print(f"  {'Company':<30} {'Opens':>7} {'Records':>8} {'Months':>7} {'Span':>10} {'First':>12} {'Last':>12} {'Status'}")
print(f"  {'-'*30} {'-'*7} {'-'*8} {'-'*7} {'-'*10} {'-'*12} {'-'*12} {'-'*12}")

for _, row in df.iterrows():
    company = row['company_name']
    months = row['months_with_data']
    span = row['date_span_days']
    opens = row['total_openings']
    
    # Check 2025-2026 monthly data
    co_monthly = monthly[monthly['company_name'] == company]
    recent_months = len(co_monthly)
    
    if recent_months >= IDEAL:
        status = "✅ SUFFICIENT"
        sufficient.append(company)
    elif recent_months >= SUFFICIENT:
        status = "⚠️  MARGINAL"
        marginal.append(company)
    else:
        status = "❌ INSUFFICIENT"
        insufficient.append(company)
    
    first = row['first_date'].strftime('%Y-%m-%d') if pd.notna(row['first_date']) else 'N/A'
    last = row['last_date'].strftime('%Y-%m-%d') if pd.notna(row['last_date']) else 'N/A'
    span_str = f"{int(span)}d" if pd.notna(span) else 'N/A'
    
    print(f"  {company:<30} {int(opens):>7} {int(row['total_records']):>8} {recent_months:>7} {span_str:>10} {first:>12} {last:>12} {status}")

# Companies in KEEP list but not in DB
keep_list = [
    'Caterpillar','Amgen','Samsung','IN Caterpillar','John Deere',
    'IN Standard Chartered','IN Amazon','Facebook','Verizon One','Microsoft',
    'IN MasterCard','IN PayPal','Wabtec Corporation','T-Mobile','PayPal',
    'IN Diageo PLC','IN Microsoft','Salesforce','eBay','Pfizer',
    'Amazon','Visa','Blue Shield of California','IN Linkedin','RELX Inc.',
    'IN Salesforce','IN Intuit','Best Buy','Oracle','GAP',
    'Catalent Corporation','Novo Nordisk','Zebra Technologies','Excellus',
    'State Street','ADVARRA, Inc.','Dell','Expedia','Rubrik',
    'Kroger','SAIC','Campbells','IHG','IN Rubrik',
    'PepsiCo Inc','IN Q2 Financials','Pearson','Diageo','Seqirus',
    'Solidigm','Daimler','GE Aerospace','Integra LifeSciences',
    'Mercedes-Benz','Red Hat','CIN7','IN PepsiCo','NAIC',
    'Blue Origin','GE Healthcare'
]
found_companies = df['company_name'].tolist()
for c in keep_list:
    if c not in found_companies:
        not_found.append(c)
        print(f"  {c:<30} {'N/A':>7} {'N/A':>8} {'0':>7} {'N/A':>10} {'N/A':>12} {'N/A':>12} ❌ NOT IN DB")

print(f"\n{'='*100}")
print(f"  SUMMARY")
print(f"{'='*100}")
print(f"  ✅ Sufficient (12+ months 2025-2026): {len(sufficient)}")
print(f"  ⚠️  Marginal (6-11 months):            {len(marginal)}")
print(f"  ❌ Insufficient (<6 months):           {len(insufficient)}")
print(f"  ❌ Not in DB at all:                   {len(not_found)}")

if insufficient or not_found:
    print(f"\n  COMPANIES NEEDING MORE DATA (career page scrape priority):")
    for c in insufficient:
        co_monthly_data = monthly[monthly['company_name'] == c]
        months_str = ', '.join(co_monthly_data['month'].tolist())
        print(f"    🔴 {c:<30} (has: {months_str})")
    for c in not_found:
        print(f"    🔴 {c:<30} (NO DATA AT ALL)")

if marginal:
    print(f"\n  COMPANIES WITH MARGINAL DATA (would benefit from career page scrape):")
    for c in marginal:
        co_monthly_data = monthly[monthly['company_name'] == c]
        months_str = ', '.join(co_monthly_data['month'].tolist())
        print(f"    🟡 {c:<30} (has: {months_str})")

# Career pages we already scraped
print(f"\n  CAREER PAGES ALREADY SCRAPED:")
scraped = ['Amazon', 'Rubrik', 'Caterpillar', 'Amgen', 'T-Mobile', 'eBay', 'Pfizer', 'Visa']
for c in scraped:
    status = "✅ SUFFICIENT" if c in sufficient else ("⚠️ MARGINAL" if c in marginal else "❌ INSUFFICIENT")
    print(f"    ✅ {c:<30} DB status: {status}")

print(f"\n  STILL NEED TO SCRAPE ({len(insufficient) + len(not_found) + len(marginal) - len([c for c in scraped if c in marginal or c in insufficient])} companies):")
need_scrape = [c for c in (insufficient + not_found + marginal) if c not in scraped]
for c in need_scrape:
    print(f"    🔲 {c}")
