"""Import issued jobs JSON files into job_records table."""
import os
import json
import glob
import mysql.connector
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST"),
    "port": int(os.getenv("MYSQL_PORT")),
    "database": os.getenv("MYSQL_DATABASE"),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
}

INSERT_SQL = """
INSERT INTO job_records (
    job_id, job_diva_no, optional_reference_no, division_id, division_name,
    primary_recruiter_id, primary_recruiter, primary_sales_id, primary_sales,
    country, company_id, company_name, contact_id, contact_first_name, contact_last_name,
    issue_date, start_date, end_date, position_type, job_status, title,
    openings, fills, city, state, zipcode,
    bill_rate_min, bill_rate_max, pay_rate_min, pay_rate_max,
    onsite_flexibility, remote_percentage
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s
) ON DUPLICATE KEY UPDATE
    job_diva_no=VALUES(job_diva_no), division_name=VALUES(division_name),
    company_name=VALUES(company_name), job_status=VALUES(job_status),
    title=VALUES(title), openings=VALUES(openings), fills=VALUES(fills),
    bill_rate_min=VALUES(bill_rate_min), bill_rate_max=VALUES(bill_rate_max),
    pay_rate_min=VALUES(pay_rate_min), pay_rate_max=VALUES(pay_rate_max)
"""


def parse_datetime(val):
    """Return datetime string or None."""
    if not val or val.strip() == "":
        return None
    return val


def safe_decimal(val):
    try:
        return float(val) if val else 0
    except (ValueError, TypeError):
        return 0


def safe_int(val):
    try:
        return int(val) if val else 0
    except (ValueError, TypeError):
        return 0


def transform(job):
    return (
        str(job.get("JOBID", "")),
        job.get("JOBDIVANO", ""),
        job.get("OPTIONALREFERENCENO", ""),
        job.get("DIVISIONID", ""),
        job.get("DIVISIONNAME", ""),
        job.get("PRIMARYRECRUITERID", ""),
        job.get("PRIMARYRECRUITER", "").strip(),
        job.get("PRIMARYSALESID", ""),
        job.get("PRIMARYSALES", "").strip(),
        job.get("COUNTRY", ""),
        job.get("COMPANYID", ""),
        job.get("COMPANYNAME", ""),
        job.get("CONTACTID", ""),
        job.get("CONTACTFIRSTNAME", ""),
        job.get("CONTACTLASTNAME", ""),
        parse_datetime(job.get("ISSUEDATE")),
        parse_datetime(job.get("STARTDATE")),
        parse_datetime(job.get("ENDDATE")),
        job.get("POSITIONTYPE", ""),
        job.get("JOBSTATUS", ""),
        job.get("TITLE", ""),
        safe_int(job.get("OPENINGS")),
        safe_int(job.get("FILLS")),
        job.get("CITY", ""),
        job.get("STATE", ""),
        job.get("ZIPCODE", ""),
        safe_decimal(job.get("BILLRATEMIN")),
        safe_decimal(job.get("BILLRATEMAX")),
        safe_decimal(job.get("PAYRATEMIN")),
        safe_decimal(job.get("PAYRATEMAX")),
        job.get("ONSITE_FLEXIBILITY", "0"),
        job.get("REMOTE_PERCENTAGE", "0"),
    )


def main():
    print("=" * 60)
    print("IMPORT ISSUED JOBS → job_records TABLE")
    print("=" * 60)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "issued_jobs")
    files = sorted(glob.glob(os.path.join(data_dir, "jobs_*.json")))
    print(f"  Found {len(files)} JSON files")

    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()

    total_inserted = 0
    for i, filepath in enumerate(files, 1):
        with open(filepath, "r", encoding="utf-8") as f:
            jobs = json.load(f)

        rows = [transform(j) for j in jobs]
        cur.executemany(INSERT_SQL, rows)
        conn.commit()
        total_inserted += len(rows)

        if i % 20 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] Imported {total_inserted} records...")

    cur.execute("SELECT COUNT(*) FROM job_records")
    count = cur.fetchone()[0]

    cur.close()
    conn.close()

    print(f"\n✓ Done! {count} records in job_records table")


if __name__ == "__main__":
    main()
