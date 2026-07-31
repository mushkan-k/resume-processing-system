"""
Fetch all issued jobs from JobDiva in 14-day intervals over the last 1.5 years.
Saves results as JSON files with max 100 jobs per file.

Output: data/issued_jobs/jobs_001.json, jobs_002.json, ...
"""
import os
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.jobdiva.com/apiv2"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "issued_jobs")
JOBS_PER_FILE = 100


def authenticate():
    client_id = os.getenv("JOBDIVA_CLIENT_ID")
    username = os.getenv("JOBDIVA_USERNAME")
    password = os.getenv("JOBDIVA_PASSWORD")
    resp = requests.get(
        f"{BASE_URL}/authenticate",
        params={"clientid": client_id, "username": username, "password": password},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.text.strip()


def fetch_chunk(token, from_date, to_date):
    """Fetch issued jobs for a 14-day window."""
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    params = {
        "fromDate": from_date.strftime("%m/%d/%Y"),
        "toDate": to_date.strftime("%m/%d/%Y"),
    }
    resp = requests.get(
        f"{BASE_URL}/bi/IssuedJobsList",
        params=params,
        headers=headers,
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    records = data.get("data", [])
    return records if isinstance(records, list) else []


def save_batch(jobs, file_num):
    """Save a batch of jobs to a numbered JSON file."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, f"jobs_{file_num:03d}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2, ensure_ascii=False)
    print(f"  Saved {len(jobs)} jobs → {path}")
    return path


def main():
    print("=" * 60)
    print("FETCH ISSUED JOBS — Last 4 Years (14-day chunks)")
    print("=" * 60)

    # Date range: 4 years ago → today
    end_date = datetime.now()
    start_date = end_date - timedelta(days=int(365 * 4))  # ~1460 days

    print(f"  Range: {start_date.strftime('%m/%d/%Y')} → {end_date.strftime('%m/%d/%Y')}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Max per file: {JOBS_PER_FILE}")
    print()

    # Authenticate
    print("Authenticating with JobDiva...")
    token = authenticate()
    print("  ✓ Authenticated\n")

    # Fetch in 14-day chunks
    all_jobs = []
    chunk_start = start_date
    chunk_num = 0

    while chunk_start < end_date:
        chunk_end = min(chunk_start + timedelta(days=14), end_date)
        chunk_num += 1
        print(f"  [{chunk_num}] {chunk_start.strftime('%m/%d/%Y')} → {chunk_end.strftime('%m/%d/%Y')} ...", end=" ")

        try:
            jobs = fetch_chunk(token, chunk_start, chunk_end)
            print(f"{len(jobs)} jobs")
            all_jobs.extend(jobs)
        except Exception as e:
            print(f"FAILED: {e}")
            # Re-authenticate in case token expired
            try:
                token = authenticate()
                jobs = fetch_chunk(token, chunk_start, chunk_end)
                print(f"  (retry) {len(jobs)} jobs")
                all_jobs.extend(jobs)
            except Exception as e2:
                print(f"  (retry failed): {e2}")

        chunk_start = chunk_end

    print(f"\n  Total jobs fetched: {len(all_jobs)}")

    # Deduplicate by job ID (in case of overlaps)
    seen = set()
    unique_jobs = []
    for job in all_jobs:
        job_id = job.get("JOBID") or job.get("ID") or json.dumps(job)
        if job_id not in seen:
            seen.add(job_id)
            unique_jobs.append(job)

    print(f"  Unique jobs: {len(unique_jobs)}")

    # Save in batches of 100
    file_num = 0
    for i in range(0, len(unique_jobs), JOBS_PER_FILE):
        file_num += 1
        batch = unique_jobs[i:i + JOBS_PER_FILE]
        save_batch(batch, file_num)

    print(f"\n✓ Done! {file_num} file(s) saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
