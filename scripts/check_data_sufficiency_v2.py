"""
Check data sufficiency per company — FIXED version.
Uses raw SQL via cursor instead of pd.read_sql to avoid %% escaping issues.
"""
import mysql.connector
from datetime import datetime

DB_CONFIG = {
    "host": "localhost", "port": 3305, "database": "resume_processing",
    "user": "resume_user", "password": "resume_password",
}

conn = mysql.connector.connect(**DB_CONFIG)
cur = conn.cursor(dictionary=True)

# Get all 61 KEEP companies data since 2025-01-01 (18 months lookback)
cur.execute("""
    SELECT company_name,
           COUNT(*) as total_records,
           SUM(openings) as total_openings,
           COUNT(DISTINCT DATE_FORMAT(issue_date, '%Y-%m')) as months_with_data,
           MIN(issue_date) as first_date,
           MAX(issue_date) as last_date,
           COUNT(DISTINCT title) as unique_titles
    FROM updated_job_records
    WHERE issue_date >= '2025-01-01'
    GROUP BY company_name
    ORDER BY total_openings DESC
""")
rows = cur.fetchall()

# Also get full history depth
cur.execute("""
    SELECT company_name,
           MIN(issue_date) as hist_first,
           COUNT(DISTINCT DATE_FORMAT(issue_date, '%Y-%m')) as hist_months
    FROM updated_job_records
    GROUP BY company_name
""")
hist_rows = {r['company_name']: r for r in cur.fetchall()}

# Monthly detail per company (2025+)
cur.execute("""
    SELECT company_name, DATE_FORMAT(issue_date, '%Y-%m') as month, 
           SUM(openings) as openings, COUNT(*) as records
    FROM updated_job_records
    WHERE issue_date >= '2025-01-01'
    GROUP BY company_name, DATE_FORMAT(issue_date, '%Y-%m')
    ORDER BY company_name, month
""")
monthly_rows = cur.fetchall()
conn.close()

# Build monthly lookup
monthly_by_company = {}
for r in monthly_rows:
    c = r['company_name']
    if c not in monthly_by_company:
        monthly_by_company[c] = []
    monthly_by_company[c].append(r)

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

# Build lookup
data_by_company = {r['company_name']: r for r in rows}

print("=" * 110)
print("  DATA SUFFICIENCY CHECK — 61 KEEP COMPANIES (2025-01 to present)")
print("=" * 110)
print(f"  Model needs: 6+ months of recent data for reliable forecasting. 12+ is ideal.\n")

SUFFICIENT = 12
MARGINAL = 6

sufficient = []
marginal = []
insufficient = []
not_found = []

print(f"  {'Company':<30} {'Opens':>7} {'Recs':>6} {'Mo(recent)':>10} {'Mo(all)':>8} {'Hist From':>12} {'Recent':>20} {'Status'}")
print(f"  {'-'*30} {'-'*7} {'-'*6} {'-'*10} {'-'*8} {'-'*12} {'-'*20} {'-'*15}")

for company in keep_list:
    if company in data_by_company:
        row = data_by_company[company]
        months_recent = row['months_with_data']
        opens = int(row['total_openings'])
        recs = int(row['total_records'])
        hist = hist_rows.get(company, {})
        hist_months = hist.get('hist_months', 0)
        hist_first = hist.get('hist_first', None)
        hist_str = hist_first.strftime('%Y-%m') if hist_first else 'N/A'
        
        # Get recent months list
        mo_list = monthly_by_company.get(company, [])
        mo_str = ','.join([m['month'][-2:] for m in mo_list])  # just month numbers
        
        if months_recent >= SUFFICIENT:
            status = "✅ SUFFICIENT"
            sufficient.append(company)
        elif months_recent >= MARGINAL:
            status = "⚠️  MARGINAL"
            marginal.append(company)
        else:
            status = "❌ NEED MORE"
            insufficient.append(company)
        
        print(f"  {company:<30} {opens:>7} {recs:>6} {months_recent:>10} {hist_months:>8} {hist_str:>12} {mo_str:>20} {status}")
    else:
        not_found.append(company)
        print(f"  {company:<30} {'—':>7} {'—':>6} {'0':>10} {'—':>8} {'—':>12} {'':>20} ❌ NOT IN DB")

print(f"\n{'='*110}")
print(f"  SUMMARY")
print(f"{'='*110}")
print(f"  ✅ Sufficient (12+ recent months):  {len(sufficient)}")
print(f"  ⚠️  Marginal (6-11 months):          {len(marginal)}")
print(f"  ❌ Insufficient (<6 months):         {len(insufficient)}")
print(f"  ❌ Not in DB:                        {len(not_found)}")

scraped = {'Amazon', 'Rubrik', 'Caterpillar', 'Amgen', 'T-Mobile', 'eBay', 'Pfizer', 'Visa'}

if insufficient or not_found:
    print(f"\n  🔴 NEED CAREER PAGE SCRAPING ({len(insufficient) + len(not_found)} companies):")
    for c in insufficient + not_found:
        already = "✅ scraped" if c in scraped else "🔲 NOT scraped"
        print(f"    {c:<30} {already}")

if marginal:
    print(f"\n  🟡 MARGINAL — would benefit from scraping ({len(marginal)} companies):")
    for c in marginal:
        already = "✅ scraped" if c in scraped else "🔲 NOT scraped"
        print(f"    {c:<30} {already}")

if sufficient:
    print(f"\n  ✅ SUFFICIENT — no scraping needed ({len(sufficient)} companies):")
    for c in sufficient:
        print(f"    {c}")
