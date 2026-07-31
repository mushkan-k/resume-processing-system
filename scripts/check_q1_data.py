"""Quick check: what Q1 2026 data exists in DB."""
import mysql.connector

conn = mysql.connector.connect(
    host='localhost', port=3305, database='resume_processing',
    user='resume_user', password='resume_password'
)
cur = conn.cursor(dictionary=True)

# Check updated_job_records for Q1
cur.execute("SELECT MIN(issue_date) as min_date, MAX(issue_date) as max_date, COUNT(*) as total FROM updated_job_records WHERE issue_date >= '2026-01-01' AND issue_date < '2026-04-01'")
print('updated_job_records Q1:', cur.fetchone())

# Check job_descriptions for Q1
cur.execute("SELECT MIN(issue_date) as min_date, MAX(issue_date) as max_date, COUNT(*) as total FROM job_descriptions WHERE issue_date >= '2026-01-01' AND issue_date < '2026-04-01'")
print('job_descriptions Q1:', cur.fetchone())

# Check demand_forecasts actuals for Q1
cur.execute("SELECT COUNT(*) as c, COUNT(DISTINCT cluster_name) as clusters FROM demand_forecasts WHERE data_type='actual' AND forecast_month >= '2026-01-01' AND forecast_month < '2026-04-01'")
print('demand_forecasts actuals Q1:', cur.fetchone())

# Overall date range in updated_job_records
cur.execute("SELECT MIN(issue_date) as min_date, MAX(issue_date) as max_date, COUNT(*) as total FROM updated_job_records")
print('updated_job_records TOTAL:', cur.fetchone())

conn.close()
