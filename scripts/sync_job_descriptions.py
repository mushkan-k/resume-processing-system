"""
Sync Job Descriptions from JobDiva API into the job_descriptions table.

This script fetches job records from JobDiva and upserts them into MySQL.
It should be run periodically (e.g. every 14 days or via scheduler).

Usage:
    python scripts/sync_job_descriptions.py
    python scripts/sync_job_descriptions.py --days 30    # fetch last 30 days
    python scripts/sync_job_descriptions.py --from-date "05/01/2026 00:00:00" --to-date "06/09/2026 00:00:00"
"""
import os
import sys
import argparse
import requests
import mysql.connector
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────
DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", "3305")),
    "database": os.getenv("MYSQL_DATABASE", "resume_processing"),
    "user": os.getenv("MYSQL_USER", "resume_user"),
    "password": os.getenv("MYSQL_PASSWORD", "resume_password"),
}

JOBDIVA_BASE_URL = "https://api.jobdiva.com/apiv2"
JOBDIVA_CLIENT_ID = os.getenv("JOBDIVA_CLIENT_ID")
JOBDIVA_USERNAME = os.getenv("JOBDIVA_USERNAME")
JOBDIVA_PASSWORD = os.getenv("JOBDIVA_PASSWORD")

DATE_FMT = "%m/%d/%Y %H:%M:%S"

UPSERT_SQL = """
INSERT INTO job_descriptions (
    job_id, job_diva_no, title, company_name, division_name,
    position_type, job_status, city, state, zipcode, country,
    address1, address2, issue_date, start_date, end_date,
    max_allowed_submittals
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
) ON DUPLICATE KEY UPDATE
    job_diva_no = VALUES(job_diva_no),
    title = VALUES(title),
    company_name = VALUES(company_name),
    division_name = VALUES(division_name),
    position_type = VALUES(position_type),
    job_status = VALUES(job_status),
    city = VALUES(city),
    state = VALUES(state),
    zipcode = VALUES(zipcode),
    country = VALUES(country),
    address1 = VALUES(address1),
    address2 = VALUES(address2),
    issue_date = VALUES(issue_date),
    start_date = VALUES(start_date),
    end_date = VALUES(end_date),
    max_allowed_submittals = VALUES(max_allowed_submittals)
"""


def authenticate():
    """Get JWT token from JobDiva."""
    if not all([JOBDIVA_CLIENT_ID, JOBDIVA_USERNAME, JOBDIVA_PASSWORD]):
        print("ERROR: JobDiva credentials not set in .env")
        sys.exit(1)

    url = f"{JOBDIVA_BASE_URL}/authenticate"
    params = {"clientid": JOBDIVA_CLIENT_ID, "username": JOBDIVA_USERNAME, "password": JOBDIVA_PASSWORD}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.text.strip()


def parse_datetime(val):
    """Parse a datetime string from JobDiva, return datetime or None."""
    if not val or val.strip() == "" or val == "Null":
        return None
    try:
        # Try ISO format first
        return datetime.fromisoformat(val.replace("Z", "+00:00")).replace(tzinfo=None)
    except (ValueError, AttributeError):
        pass
    try:
        return datetime.strptime(val, "%m/%d/%Y %H:%M:%S")
    except (ValueError, AttributeError):
        return None


def safe_int(val):
    """Safely convert to int."""
    try:
        return int(val) if val else 0
    except (ValueError, TypeError):
        return 0


def fetch_jobs(token, from_dt, to_dt):
    """Fetch jobs from JobDiva in 14-day chunks."""
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    all_records = []
    chunk_start = from_dt

    while chunk_start < to_dt:
        chunk_end = min(chunk_start + timedelta(days=14), to_dt)
        params = {
            "fromDate": chunk_start.strftime(DATE_FMT),
            "toDate": chunk_end.strftime(DATE_FMT),
        }

        print(f"  Fetching: {chunk_start.strftime('%m/%d/%Y')} → {chunk_end.strftime('%m/%d/%Y')}...", end=" ")

        try:
            resp = requests.get(
                f"{JOBDIVA_BASE_URL}/bi/NewUpdatedJobRecords",
                params=params,
                headers=headers,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            records = data.get("data", [])
            if isinstance(records, list):
                all_records.extend(records)
                print(f"{len(records)} records")
            else:
                print("0 records")
        except Exception as e:
            print(f"ERROR: {e}")

        chunk_start = chunk_end

    return all_records


def upsert_to_db(records):
    """Insert/update records into job_descriptions table."""
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()

    inserted = 0
    for r in records:
        values = (
            str(r.get("JOBID", "")),
            r.get("JOBDIVANO"),
            r.get("TITLE"),
            r.get("COMPANYNAME"),
            r.get("DIVISIONNAME"),
            r.get("POSITIONTYPE"),
            r.get("JOBSTATUS"),
            r.get("CITY"),
            r.get("STATE"),
            r.get("ZIPCODE"),
            r.get("COUNTRY"),
            r.get("ADDRESS1"),
            r.get("ADDRESS2"),
            parse_datetime(r.get("ISSUEDATE")),
            parse_datetime(r.get("STARTDATE")),
            parse_datetime(r.get("ENDDATE")),
            safe_int(r.get("MAXALLOWEDSUBMITTALS")),
        )
        try:
            cur.execute(UPSERT_SQL, values)
            inserted += 1
        except Exception as e:
            print(f"  Error inserting job {r.get('JOBID')}: {e}")

    conn.commit()
    cur.close()
    conn.close()
    return inserted


def main():
    parser = argparse.ArgumentParser(description="Sync Job Descriptions from JobDiva to DB")
    parser.add_argument("--days", type=int, default=14, help="Fetch last N days (default: 14)")
    parser.add_argument("--from-date", type=str, help="Start date (MM/DD/YYYY HH:MM:SS)")
    parser.add_argument("--to-date", type=str, help="End date (MM/DD/YYYY HH:MM:SS)")
    args = parser.parse_args()

    print("=" * 60)
    print("SYNC JOB DESCRIPTIONS FROM JOBDIVA")
    print("=" * 60)

    # Determine date range
    now = datetime.now()
    if args.from_date:
        from_dt = datetime.strptime(args.from_date, DATE_FMT)
    else:
        from_dt = now - timedelta(days=args.days)

    if args.to_date:
        to_dt = datetime.strptime(args.to_date, DATE_FMT)
    else:
        to_dt = now

    print(f"  Date range: {from_dt.strftime('%m/%d/%Y')} → {to_dt.strftime('%m/%d/%Y')}")
    print(f"  Days: {(to_dt - from_dt).days}")
    print()

    # Authenticate
    print("  Authenticating with JobDiva...")
    token = authenticate()
    print("  ✓ Authenticated")
    print()

    # Fetch
    print("  Fetching job records...")
    records = fetch_jobs(token, from_dt, to_dt)
    print(f"\n  Total fetched: {len(records)} records")
    print()

    if not records:
        print("  No records to sync.")
        return

    # Upsert
    print("  Upserting to job_descriptions table...")
    count = upsert_to_db(records)
    print(f"  ✓ Upserted {count} records")
    print()
    print("=" * 60)
    print("SYNC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
