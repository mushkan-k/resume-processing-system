"""
JobDiva JD + Resume Pipeline
==============================
Chain: NewUpdatedJobRecords → JOBID → JobApplicantsDetail → RESUMEID → ResumeDetail

This script:
1. Fetches job records (JDs) for a date range
2. For each job, fetches linked applicants
3. For each applicant, fetches their resume detail
4. Outputs a structured dataset linking JDs ↔ Resumes

Usage:
    python jobdiva_jd_resume_pipeline.py --from-date 2026-01-01 --to-date 2026-06-30
    python jobdiva_jd_resume_pipeline.py --from-date 2026-01-01 --to-date 2026-06-30 --max-jobs 10  # test mode
"""
import os
import sys
import json
import csv
import requests
import logging
import argparse
from datetime import datetime
from time import sleep
from typing import Optional, List, Dict, Any

# Load .env file
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

BASE_URL = "https://api.jobdiva.com/apiv2"
RETRY_LIMIT = 3
RETRY_DELAY = 5
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'jobdiva_extracts')


# ═══════════════════════════════════════════════════════════════════════
# AUTH
# ═══════════════════════════════════════════════════════════════════════

def authenticate() -> str:
    """Authenticate with JobDiva and return JWT token."""
    client_id = os.getenv("JOBDIVA_CLIENT_ID")
    username = os.getenv("JOBDIVA_USERNAME")
    password = os.getenv("JOBDIVA_PASSWORD")

    if not all([client_id, username, password]):
        raise RuntimeError(
            "Missing JobDiva credentials. Set env vars:\n"
            "  JOBDIVA_CLIENT_ID\n"
            "  JOBDIVA_USERNAME\n"
            "  JOBDIVA_PASSWORD"
        )

    resp = requests.get(f"{BASE_URL}/authenticate", params={
        "clientid": client_id,
        "username": username,
        "password": password,
    }, timeout=30)
    resp.raise_for_status()
    token = resp.text.strip().strip('"')

    if token.count(".") != 2:
        raise RuntimeError(f"Invalid JWT token received (length={len(token)})")

    logger.info("✅ Authenticated with JobDiva")
    return token


def api_get(token: str, endpoint: str, params: dict, timeout: int = 60) -> Any:
    """Make authenticated GET request with retry logic."""
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    url = f"{BASE_URL}/{endpoint}"

    for attempt in range(RETRY_LIMIT):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
            if resp.status_code == 401:
                # Token expired, re-auth
                logger.warning("Token expired, re-authenticating...")
                token = authenticate()
                headers["Authorization"] = f"Bearer {token}"
                continue
            resp.raise_for_status()
            return resp.json(), token
        except requests.exceptions.Timeout:
            logger.warning(f"Timeout on {endpoint}, attempt {attempt+1}/{RETRY_LIMIT}")
            sleep(RETRY_DELAY * (attempt + 1))
        except Exception as e:
            if attempt == RETRY_LIMIT - 1:
                raise
            logger.warning(f"Request failed ({e}), retrying {attempt+1}/{RETRY_LIMIT}...")
            sleep(RETRY_DELAY)

    return None, token


def convert_date(date_str: str) -> str:
    """Convert YYYY-MM-DD to JobDiva format MM/DD/YYYY HH:MM:SS."""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.strftime("%m/%d/%Y 00:00:00")


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Fetch Job Records (JDs)
# ═══════════════════════════════════════════════════════════════════════

def fetch_job_records(token: str, from_date: str, to_date: str) -> tuple:
    """
    Fetch job records from /apiv2/bi/NewUpdatedJobRecords.
    API has a 14-day limit, so we chunk the date range automatically.
    Returns list of job dicts and (possibly refreshed) token.
    """
    logger.info(f"📋 Fetching job records: {from_date} → {to_date}")

    from datetime import timedelta
    start = datetime.strptime(from_date, "%Y-%m-%d")
    end = datetime.strptime(to_date, "%Y-%m-%d")
    
    all_jobs = []
    chunk_start = start
    
    while chunk_start < end:
        chunk_end = min(chunk_start + timedelta(days=13), end)  # 14-day max window
        
        logger.info(f"  Chunk: {chunk_start.strftime('%Y-%m-%d')} → {chunk_end.strftime('%Y-%m-%d')}")
        
        data, token = api_get(token, "bi/NewUpdatedJobRecords", {
            "fromDate": chunk_start.strftime("%m/%d/%Y 00:00:00"),
            "toDate": chunk_end.strftime("%m/%d/%Y 23:59:59"),
        }, timeout=120)

        if data:
            # Handle response format
            if isinstance(data, dict):
                msg = data.get("message", "")
                records = data.get("data", [])
                if isinstance(records, list):
                    all_jobs.extend(records)
                    logger.info(f"    → {len(records)} jobs")
                elif isinstance(records, dict) and not records:
                    logger.info(f"    → 0 jobs")
            elif isinstance(data, list):
                all_jobs.extend(data)
                logger.info(f"    → {len(data)} jobs")
        
        chunk_start = chunk_end + timedelta(days=1)
        sleep(0.5)  # Rate limit between chunks
    
    logger.info(f"  Total: {len(all_jobs):,} job records")

    # Extract key fields
    parsed_jobs = []
    for job in all_jobs:
        parsed_jobs.append({
            'JOBID': job.get('JOBID') or job.get('JOB_ID') or job.get('ID'),
            'JOBDIVANO': job.get('JOBDIVANO') or job.get('JOB_DIVA_NO'),
            'TITLE': job.get('TITLE') or job.get('JOBTITLE'),
            'COMPANY': job.get('COMPANYNAME') or job.get('COMPANY_NAME') or job.get('COMPANY'),
            'STATUS': job.get('JOBSTATUS') or job.get('JOB_STATUS') or job.get('STATUS'),
            'ISSUE_DATE': job.get('ISSUEDATE') or job.get('ISSUE_DATE'),
            'CITY': job.get('CITY'),
            'STATE': job.get('STATE'),
            'COUNTRY': job.get('COUNTRY'),
            'OPENINGS': job.get('OPENINGS', 1),
            'FILLS': job.get('FILLS', 0),
            'SKILLS': job.get('SKILLS', ''),
            'BILL_RATE_MIN': job.get('BILLRATEMIN') or job.get('BILL_RATE_MIN'),
            'BILL_RATE_MAX': job.get('BILLRATEMAX') or job.get('BILL_RATE_MAX'),
            '_raw': job,  # Keep raw for debugging
        })

    return parsed_jobs, token


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Fetch Applicants per Job
# ═══════════════════════════════════════════════════════════════════════

def fetch_job_applicants(token: str, job_id: str) -> tuple:
    """
    Fetch applicants for a job from /apiv2/bi/JobApplicantsDetail.
    Returns list of applicant dicts and (possibly refreshed) token.
    """
    data, token = api_get(token, "bi/JobApplicantsDetail", {
        "jobId": str(job_id),
    }, timeout=60)

    if not data:
        return [], token

    if isinstance(data, dict) and "data" in data:
        applicants = data["data"]
    elif isinstance(data, list):
        applicants = data
    else:
        return [], token

    parsed = []
    for app in applicants:
        parsed.append({
            'CANDIDATEID': app.get('CANDIDATEID') or app.get('CANDIDATE_ID'),
            'RESUMEID': app.get('RESUMEID') or app.get('RESUME_ID'),
            'FIRSTNAME': app.get('FIRSTNAME') or app.get('FIRST_NAME', ''),
            'LASTNAME': app.get('LASTNAME') or app.get('LAST_NAME', ''),
            'STATUS': app.get('STATUS') or app.get('APPLICANT_STATUS', ''),
            'APPLY_DATE': app.get('APPLYDATE') or app.get('APPLY_DATE', ''),
            'SUBMISSION_DATE': app.get('SUBMISSIONDATE') or app.get('SUBMISSION_DATE', ''),
            '_raw': app,
        })

    return parsed, token


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Fetch Resume Detail
# ═══════════════════════════════════════════════════════════════════════

def fetch_resume_detail(token: str, resume_id: str) -> tuple:
    """
    Fetch resume content from /apiv2/bi/ResumeDetail.
    Returns resume dict and (possibly refreshed) token.
    """
    data, token = api_get(token, "bi/ResumeDetail", {
        "resumeId": str(resume_id),
    }, timeout=90)

    if not data:
        return None, token

    if isinstance(data, dict) and "data" in data:
        recs = data["data"]
    elif isinstance(data, list):
        recs = data
    else:
        return None, token

    if not recs:
        return None, token

    rec = recs[0] if isinstance(recs, list) else recs
    return {
        'RESUMEID': resume_id,
        'PLAINTEXT': rec.get('PLAINTEXT', ''),
        'FILETYPE': rec.get('FILETYPE', ''),
        'HAS_BASE64': bool(rec.get('FILECONTENT_BASE64ENCODED', '')),
        'DATE_CREATED': rec.get('DATECREATED', ''),
    }, token


# ═══════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════

def run_pipeline(from_date: str, to_date: str, max_jobs: int = None, skip_resumes: bool = False, max_applicants_per_job: int = 50):
    """
    Full pipeline:
      NewUpdatedJobRecords → JOBID → JobApplicantsDetail → RESUMEID → ResumeDetail
    """
    print("=" * 70)
    print("  JOBDIVA JD + RESUME EXTRACTION PIPELINE")
    print(f"  Date range: {from_date} → {to_date}")
    print(f"  Max jobs: {max_jobs or 'ALL'}")
    print("=" * 70)

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Authenticate ──
    token = authenticate()

    # ── Step 2: Fetch all job records (JDs) ──
    jobs, token = fetch_job_records(token, from_date, to_date)
    if not jobs:
        print("  ❌ No jobs found. Check date range.")
        return

    if max_jobs:
        jobs = jobs[:max_jobs]
        print(f"  ⚠️  Limited to {max_jobs} jobs (test mode)")

    print(f"\n  📋 Processing {len(jobs)} jobs...")

    # ── Step 3: For each job, fetch applicants + resumes ──
    results = []
    total_applicants = 0
    total_resumes = 0

    for i, job in enumerate(jobs, 1):
        job_id = job['JOBID']
        if not job_id:
            continue

        # Progress
        logger.info(f"  [{i}/{len(jobs)}] Job {job_id} — {job['TITLE']} ({job['COMPANY']})")

        # Fetch applicants for this job
        applicants, token = fetch_job_applicants(token, job_id)
        total_applicants += len(applicants)
        logger.info(f"    → {len(applicants)} applicants")

        # Cap applicants per job if explicitly set
        if max_applicants_per_job and len(applicants) > max_applicants_per_job and not skip_resumes:
            logger.warning(f"    ⚠️  Capping to {max_applicants_per_job} applicants (had {len(applicants)})")
            applicants_to_fetch = applicants[:max_applicants_per_job]
        else:
            applicants_to_fetch = applicants

        if not applicants:
            # Job with no applicants — still record the JD
            results.append({
                'job_id': job_id,
                'job_title': job['TITLE'],
                'company': job['COMPANY'],
                'status': job['STATUS'],
                'issue_date': job['ISSUE_DATE'],
                'city': job['CITY'],
                'state': job['STATE'],
                'country': job['COUNTRY'],
                'openings': job['OPENINGS'],
                'fills': job['FILLS'],
                'skills': job['SKILLS'],
                'bill_rate_min': job['BILL_RATE_MIN'],
                'bill_rate_max': job['BILL_RATE_MAX'],
                'applicant_count': 0,
                'candidate_id': None,
                'resume_id': None,
                'candidate_name': None,
                'applicant_status': None,
                'resume_plaintext': None,
                'resume_filetype': None,
            })
            continue

        # For each applicant, optionally fetch resume
        for j, app in enumerate(applicants_to_fetch, 1):
            resume_id = app.get('RESUMEID')
            resume_data = None

            if resume_id and not skip_resumes:
                if j == 1 or j == len(applicants_to_fetch) or j % 10 == 0:
                    logger.info(f"    Applicant {j}/{len(applicants_to_fetch)} — resume {resume_id}")
                try:
                    resume_data, token = fetch_resume_detail(token, resume_id)
                    if resume_data:
                        total_resumes += 1
                except Exception as e:
                    logger.warning(f"    Resume {resume_id} failed: {e}")
                # Rate limiting — be nice to the API
                sleep(0.2)

            results.append({
                'job_id': job_id,
                'job_title': job['TITLE'],
                'company': job['COMPANY'],
                'status': job['STATUS'],
                'issue_date': job['ISSUE_DATE'],
                'city': job['CITY'],
                'state': job['STATE'],
                'country': job['COUNTRY'],
                'openings': job['OPENINGS'],
                'fills': job['FILLS'],
                'skills': job['SKILLS'],
                'bill_rate_min': job['BILL_RATE_MIN'],
                'bill_rate_max': job['BILL_RATE_MAX'],
                'applicant_count': len(applicants),
                'candidate_id': app.get('CANDIDATEID'),
                'resume_id': resume_id,
                'candidate_name': f"{app.get('FIRSTNAME', '')} {app.get('LASTNAME', '')}".strip(),
                'applicant_status': app.get('STATUS'),
                'resume_plaintext': resume_data.get('PLAINTEXT', '')[:500] if resume_data else None,
                'resume_filetype': resume_data.get('FILETYPE') if resume_data else None,
            })

        # Rate limiting between jobs
        sleep(0.3)

    # ── Step 4: Save results ──
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Save as JSON (full detail)
    json_path = os.path.join(OUTPUT_DIR, f'jd_resume_extract_{timestamp}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)

    # Save as CSV (flat, no resume text — for quick browsing)
    csv_path = os.path.join(OUTPUT_DIR, f'jd_resume_extract_{timestamp}.csv')
    csv_fields = [k for k in results[0].keys() if k != 'resume_plaintext'] if results else []
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)

    # Summary
    unique_jobs = len(set(r['job_id'] for r in results if r['job_id']))
    unique_candidates = len(set(r['candidate_id'] for r in results if r['candidate_id']))

    print(f"\n{'=' * 70}")
    print(f"  PIPELINE COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Jobs processed:       {unique_jobs:,}")
    print(f"  Total applicants:     {total_applicants:,}")
    print(f"  Resumes fetched:      {total_resumes:,}")
    print(f"  Unique candidates:    {unique_candidates:,}")
    print(f"  Output rows:          {len(results):,}")
    print(f"\n  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")
    print(f"{'=' * 70}")

    return results


def main():
    parser = argparse.ArgumentParser(description='JobDiva JD + Resume Pipeline')
    parser.add_argument('--from-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--to-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--max-jobs', type=int, default=None, help='Max jobs to process (for testing)')
    parser.add_argument('--max-applicants-per-job', type=int, default=None, help='Max applicants per job to fetch resumes for (default: ALL)')
    parser.add_argument('--skip-resumes', action='store_true', help='Skip resume detail fetch (just JDs + applicant links)')
    args = parser.parse_args()

    run_pipeline(args.from_date, args.to_date, args.max_jobs, args.skip_resumes, args.max_applicants_per_job)


if __name__ == '__main__':
    main()
