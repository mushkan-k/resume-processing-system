"""
Import updated job records JSON files into updated_job_records table.
Reads from data/updated_jobs/jobs_*.json and upserts into MySQL.
"""
import os
import json
import glob
import mysql.connector
from datetime import datetime

DB_CONFIG = {
    "host": "127.0.0.1",
    "port": 3305,
    "database": "resume_processing",
    "user": "resume_user",
    "password": "resume_password",
}

INSERT_SQL = """
INSERT INTO updated_job_records (
    job_id, job_diva_no, optional_reference_no, division_id, division_name,
    primary_recruiter_id, primary_sales_id, primary_owner_id, updated_by, user_id,
    msp_id, msp_name, country, company_id, company_name,
    contact_id, contact_first_name, contact_last_name,
    issue_date, start_date, end_date, date_updated, date_status_updated,
    date_user_field_updated, submittal_due,
    position_type, job_status, title, openings, fills,
    max_allowed_submittals, priority, city, state, zipcode,
    address1, address2,
    bill_rate_min, bill_rate_max, bill_rate_currency, bill_frequency,
    pay_rate_min, pay_rate_max, pay_frequency, currency,
    fee, fee_type,
    onsite_flexibility, remote_percentage, post_to_portal,
    secondary_division, skills, experience_level,
    vms_website, facility, bls_occupation, bls_occupation_id,
    eeoc_federal_sector_occupation
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
    %s, %s, %s, %s, %s, %s, %s, %s
) ON DUPLICATE KEY UPDATE
    job_diva_no=VALUES(job_diva_no),
    division_name=VALUES(division_name),
    company_name=VALUES(company_name),
    job_status=VALUES(job_status),
    title=VALUES(title),
    openings=VALUES(openings),
    fills=VALUES(fills),
    date_updated=VALUES(date_updated),
    date_status_updated=VALUES(date_status_updated),
    skills=VALUES(skills),
    experience_level=VALUES(experience_level),
    bill_rate_min=VALUES(bill_rate_min),
    bill_rate_max=VALUES(bill_rate_max),
    pay_rate_min=VALUES(pay_rate_min),
    pay_rate_max=VALUES(pay_rate_max)
"""


def parse_datetime(val):
    """Parse datetime string, return None for invalid/empty values."""
    if not val or val.strip() == "" or "1969-12-31" in val:
        return None
    try:
        return datetime.fromisoformat(val.replace("T", " ").split(".")[0])
    except (ValueError, TypeError):
        return None


def safe_decimal(val):
    try:
        return float(val) if val and val.strip() else 0
    except (ValueError, TypeError):
        return 0


def safe_int(val):
    try:
        return int(val) if val and str(val).strip() else 0
    except (ValueError, TypeError):
        return 0


def transform(job):
    """Transform a JSON job record into a tuple for INSERT."""
    return (
        str(job.get("JOBID", "")),
        job.get("JOBDIVANO", ""),
        job.get("OPTIONALREFERENCENO", ""),
        job.get("DIVISIONID", ""),
        job.get("DIVISIONNAME", ""),
        job.get("PRIMARYRECRUITERID", ""),
        job.get("PRIMARYSALESID", ""),
        job.get("PRIMARYOWNERID", ""),
        job.get("UPDATEDBY", ""),
        job.get("USERID", ""),
        job.get("MSPID", ""),
        job.get("MSPNAME", ""),
        job.get("COUNTRY", ""),
        job.get("COMPANYID", ""),
        job.get("COMPANYNAME", ""),
        job.get("CONTACTID", ""),
        job.get("CONTACTFIRSTNAME", ""),
        job.get("CONTACTLASTNAME", ""),
        parse_datetime(job.get("ISSUEDATE")),
        parse_datetime(job.get("STARTDATE")),
        parse_datetime(job.get("ENDDATE")),
        parse_datetime(job.get("DATEUPDATED")),
        parse_datetime(job.get("DATESTATUSUPDATED")),
        parse_datetime(job.get("DATEUSERFIELDUPDATED")),
        parse_datetime(job.get("SUBMITTAL_DUE")),
        job.get("POSITIONTYPE", ""),
        job.get("JOBSTATUS", ""),
        job.get("TITLE", ""),
        safe_int(job.get("OPENINGS")),
        safe_int(job.get("FILLS")),
        safe_int(job.get("MAXALLOWEDSUBMITTALS")),
        safe_int(job.get("PRIORITY")),
        job.get("CITY", ""),
        job.get("STATE", ""),
        job.get("ZIPCODE", ""),
        job.get("ADDRESS1", ""),
        job.get("ADDRESS2", ""),
        safe_decimal(job.get("BILLRATEMIN")),
        safe_decimal(job.get("BILLRATEMAX")),
        job.get("BILLRATE_CURRENCY", ""),
        job.get("BILLFREQUENCY", ""),
        safe_decimal(job.get("PAYRATEMIN")),
        safe_decimal(job.get("PAYRATEMAX")),
        job.get("PAYFREQUENCY", ""),
        job.get("CURRENCY", ""),
        safe_decimal(job.get("FEE")),
        job.get("FEE_TYPE", ""),
        job.get("ONSITE_FLEXIBILITY", ""),
        job.get("REMOTE_PERCENTAGE", ""),
        job.get("POSTTOPORTAL", ""),
        job.get("SECONDARY_DIVISION", ""),
        job.get("SKILLS", ""),
        job.get("EXPERIENCE_LEVEL", ""),
        job.get("VMS_WEBSITE", ""),
        job.get("FACILITY", ""),
        job.get("BLS_OCCUPATION", ""),
        job.get("BLS_OCCUPATION_ID", ""),
        job.get("EEOC_FEDERAL_SECTOR_OCCUPATION", ""),
    )


def main():
    print("=" * 60)
    print("IMPORT UPDATED JOBS → updated_job_records TABLE")
    print("=" * 60)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "updated_jobs")
    files = sorted(glob.glob(os.path.join(data_dir, "jobs_*.json")))
    print(f"  Found {len(files)} JSON files in {data_dir}")

    if not files:
        print("  No files to import!")
        return

    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Get current count
    cur.execute("SELECT COUNT(*) FROM updated_job_records")
    before_count = cur.fetchone()[0]
    print(f"  Current records in table: {before_count:,}")

    total_processed = 0
    total_jobs_in_files = 0
    errors = 0

    for i, filepath in enumerate(files, 1):
        with open(filepath, "r", encoding="utf-8") as f:
            jobs = json.load(f)

        total_jobs_in_files += len(jobs)
        rows = []
        for job in jobs:
            try:
                rows.append(transform(job))
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"  ⚠️ Transform error: {e}")

        if rows:
            try:
                cur.executemany(INSERT_SQL, rows)
                conn.commit()
                total_processed += len(rows)
            except Exception as e:
                print(f"  ❌ Insert error on file {filepath}: {e}")
                conn.rollback()
                errors += 1

        if i % 25 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}] Processed {total_processed:,} records...")

    # Get final count
    cur.execute("SELECT COUNT(*) FROM updated_job_records")
    after_count = cur.fetchone()[0]

    # Get date range
    cur.execute("SELECT MIN(issue_date), MAX(issue_date) FROM updated_job_records")
    date_range = cur.fetchone()

    cur.close()
    conn.close()

    print(f"\n{'='*60}")
    print(f"  IMPORT COMPLETE")
    print(f"{'='*60}")
    print(f"  Files processed: {len(files)}")
    print(f"  Records in files: {total_jobs_in_files:,}")
    print(f"  Records imported/updated: {total_processed:,}")
    print(f"  Errors: {errors}")
    print(f"  Table before: {before_count:,}")
    print(f"  Table after: {after_count:,}")
    print(f"  New records added: {after_count - before_count:,}")
    print(f"  Date range: {date_range[0]} → {date_range[1]}")


if __name__ == "__main__":
    main()
