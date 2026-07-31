"""
Fetch Q1 2026 (Jan 1 - Mar 31) actual openings from JobDiva NewUpdatedJobRecords API.
Inserts into updated_job_records table and classifies into role clusters.
"""
import os
import sys
import requests
import mysql.connector
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# ─── Config ───
JOBDIVA_BASE_URL = "https://api.jobdiva.com/apiv2"
JOBDIVA_CLIENT_ID = os.getenv("JOBDIVA_CLIENT_ID")
JOBDIVA_USERNAME = os.getenv("JOBDIVA_USERNAME")
JOBDIVA_PASSWORD = os.getenv("JOBDIVA_PASSWORD")

DB_CONFIG = {
    "host": "localhost",
    "port": 3305,
    "database": "resume_processing",
    "user": "resume_user",
    "password": "resume_password",
}

DATE_FMT = "%m/%d/%Y %H:%M:%S"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

# Q1 2026 date range
FROM_DATE = datetime(2026, 1, 1)
TO_DATE = datetime(2026, 4, 1)  # exclusive


def authenticate():
    """Get JWT token from JobDiva."""
    resp = requests.get(
        f"{JOBDIVA_BASE_URL}/authenticate",
        params={"clientid": JOBDIVA_CLIENT_ID, "username": JOBDIVA_USERNAME, "password": JOBDIVA_PASSWORD},
        timeout=30,
    )
    resp.raise_for_status()
    token = resp.text.strip()
    if token.count(".") != 2:
        raise RuntimeError("Invalid JWT token")
    return token


def fetch_jobs_chunked(token, from_dt, to_dt):
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
        print(f"  {chunk_start.strftime('%m/%d/%Y')} → {chunk_end.strftime('%m/%d/%Y')}...", end=" ")

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
                print("0 records (not a list)")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 401:
                print("TOKEN EXPIRED - re-authenticating...")
                token = authenticate()
                headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
                continue  # retry same chunk
            print(f"ERROR: {e}")
        except Exception as e:
            print(f"ERROR: {e}")

        chunk_start = chunk_end

    return all_records


def parse_datetime(val):
    if not val or val.strip() == "" or val == "Null":
        return None
    try:
        return datetime.fromisoformat(val.replace("Z", "+00:00")).replace(tzinfo=None)
    except (ValueError, AttributeError):
        pass
    try:
        return datetime.strptime(val, "%m/%d/%Y %H:%M:%S")
    except (ValueError, AttributeError):
        return None


def safe_int(val):
    try:
        return int(val) if val else 0
    except (ValueError, TypeError):
        return 0


def upsert_records(records):
    """Upsert fetched records into updated_job_records."""
    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Check if table exists
    cur.execute("SHOW TABLES LIKE 'updated_job_records'")
    if not cur.fetchone():
        print("  ERROR: updated_job_records table not found!")
        conn.close()
        return 0

    upsert_sql = """
    INSERT INTO updated_job_records (
        job_id, job_diva_no, title, company_name, division_name,
        position_type, job_status, city, state, zipcode, country,
        address1, address2, issue_date, start_date, end_date,
        openings, fills
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    ) ON DUPLICATE KEY UPDATE
        title = VALUES(title),
        company_name = VALUES(company_name),
        division_name = VALUES(division_name),
        position_type = VALUES(position_type),
        job_status = VALUES(job_status),
        city = VALUES(city),
        state = VALUES(state),
        zipcode = VALUES(zipcode),
        country = VALUES(country),
        issue_date = VALUES(issue_date),
        start_date = VALUES(start_date),
        end_date = VALUES(end_date),
        openings = VALUES(openings),
        fills = VALUES(fills)
    """

    inserted = 0
    errors = 0
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
            safe_int(r.get("OPENINGS", r.get("MAXALLOWEDSUBMITTALS", 1))),
            safe_int(r.get("FILLS", 0)),
        )
        try:
            cur.execute(upsert_sql, values)
            inserted += 1
        except Exception as e:
            errors += 1
            if errors <= 3:
                print(f"    Error: {e}")

    conn.commit()
    cur.close()
    conn.close()
    return inserted


def classify_and_summarize():
    """Load Q1 data from DB, classify into clusters, show summary."""
    # Load title_to_cluster mapping
    ttc = pd.read_pickle(os.path.join(DATA_DIR, 'title_to_cluster.pkl'))
    title_to_role = dict(zip(ttc['raw_title'], ttc['role_cluster']))
    print(f"\n  Cluster mapping: {len(title_to_role)} titles → {len(set(title_to_role.values()))} clusters")

    # Read Q1 from DB
    conn = mysql.connector.connect(**DB_CONFIG)
    df = pd.read_sql("""
        SELECT job_id, title, company_name, country, state, city, 
               issue_date, openings, fills, role_cluster
        FROM updated_job_records 
        WHERE issue_date >= '2026-01-01' AND issue_date < '2026-04-01'
    """, conn)
    conn.close()

    print(f"  Q1 2026 records in DB: {len(df)}")
    print(f"  Unique titles: {df['title'].nunique()}")
    print(f"  Unique companies: {df['company_name'].nunique()}")
    print(f"  Total openings: {df['openings'].sum()}")

    # Classify
    def get_cluster(row):
        if pd.notna(row['role_cluster']) and row['role_cluster']:
            return row['role_cluster']
        title = row['title']
        if title in title_to_role:
            return title_to_role[title]
        # Partial match
        title_lower = title.lower() if title else ''
        for raw_t, cluster in title_to_role.items():
            if raw_t.lower() in title_lower or title_lower in raw_t.lower():
                return cluster
        return None

    df['cluster'] = df.apply(get_cluster, axis=1)
    mapped = df[df['cluster'].notna()]
    unmapped = df[df['cluster'].isna()]

    print(f"\n  Mapped to clusters: {len(mapped)} ({len(mapped)/len(df)*100:.1f}%)")
    print(f"  Unmapped: {len(unmapped)} ({len(unmapped)/len(df)*100:.1f}%)")

    # Summary by cluster
    summary = mapped.groupby('cluster').agg(
        total_openings=('openings', 'sum'),
        job_count=('job_id', 'count'),
        companies=('company_name', 'nunique'),
    ).sort_values('total_openings', ascending=False)

    print(f"\n{'='*70}")
    print(f"  Q1 2026 ACTUAL OPENINGS - TOP 25 CLUSTERS")
    print(f"{'='*70}")
    print(f"  {'Cluster':<40} {'Openings':>10} {'Jobs':>8} {'Companies':>10}")
    print(f"  {'-'*40} {'-'*10} {'-'*8} {'-'*10}")
    for cluster, row in summary.head(25).iterrows():
        print(f"  {cluster:<40} {row['total_openings']:>10} {row['job_count']:>8} {row['companies']:>10}")

    print(f"\n  TOTALS:")
    print(f"    Clusters: {len(summary)}")
    print(f"    Total openings: {summary['total_openings'].sum()}")
    print(f"    Total job records: {summary['job_count'].sum()}")

    # Monthly breakdown
    mapped['month'] = mapped['issue_date'].dt.to_period('M')
    monthly = mapped.groupby('month')['openings'].sum()
    print(f"\n  MONTHLY BREAKDOWN:")
    for month, opens in monthly.items():
        print(f"    {month}: {opens} openings")

    return summary


def main():
    print("=" * 70)
    print("  FETCH Q1 2026 ACTUAL OPENINGS FROM JOBDIVA")
    print("=" * 70)
    print(f"  Date range: {FROM_DATE.strftime('%m/%d/%Y')} → {TO_DATE.strftime('%m/%d/%Y')}")
    print()

    # Step 1: Authenticate
    print("  [1] Authenticating with JobDiva...")
    token = authenticate()
    print("  ✓ Authenticated\n")

    # Step 2: Fetch Q1 records
    print("  [2] Fetching Q1 2026 job records (14-day chunks)...")
    records = fetch_jobs_chunked(token, FROM_DATE, TO_DATE)
    print(f"\n  Total fetched from API: {len(records)} records")

    # Deduplicate by JOBID
    seen = {}
    for r in records:
        job_id = r.get("JOBID", "")
        seen[job_id] = r
    unique = list(seen.values())
    print(f"  Unique jobs (deduplicated): {len(unique)}")

    # Step 3: Upsert to DB
    if unique:
        print(f"\n  [3] Upserting {len(unique)} records to updated_job_records...")
        count = upsert_records(unique)
        print(f"  ✓ Upserted {count} records")
    else:
        print("\n  [3] No records to upsert")

    # Step 4: Classify and show summary
    print(f"\n  [4] Classifying into role clusters and summarizing...")
    summary = classify_and_summarize()

    print(f"\n{'='*70}")
    print("  DONE — Q1 2026 actuals fetched and classified")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
