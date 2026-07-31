"""Job Descriptions endpoint — reads from job_descriptions table (synced from JobDiva)."""
import os
import requests
from datetime import datetime, timedelta
from typing import Optional
from fastapi import APIRouter, Query, HTTPException
from api.database import db_pool
from api.interviewer_matching import find_interviewers, get_employee_skill_profile

router = APIRouter(prefix="/api/jds", tags=["Job Descriptions"])

BASE_URL = "https://api.jobdiva.com/apiv2"


@router.get("")
def get_job_descriptions(
    search: Optional[str] = Query(None, description="Search across title, company, city"),
    job_status: Optional[str] = Query(None, description="Filter by job status e.g. OPEN, CLOSED"),
    position_type: Optional[str] = Query(None, description="Filter by position type"),
    company: Optional[str] = Query(None, description="Filter by company name"),
    country: Optional[str] = Query(None, description="Filter by country code"),
    state: Optional[str] = Query(None, description="Filter by state"),
    city: Optional[str] = Query(None, description="Filter by city"),
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=500),
):
    """Fetch job descriptions from the database (synced from JobDiva)."""
    conn = db_pool.get_connection()
    try:
        cur = conn.cursor(dictionary=True)

        # Build WHERE clause
        conditions = []
        params = []

        if search:
            conditions.append("(title LIKE %s OR company_name LIKE %s OR city LIKE %s OR job_diva_no LIKE %s)")
            search_param = f"%{search}%"
            params.extend([search_param, search_param, search_param, search_param])

        if job_status:
            conditions.append("job_status = %s")
            params.append(job_status.upper())

        if position_type:
            conditions.append("position_type = %s")
            params.append(position_type)

        if company:
            conditions.append("company_name LIKE %s")
            params.append(f"%{company}%")

        if country:
            conditions.append("country = %s")
            params.append(country.upper())

        if state:
            conditions.append("state = %s")
            params.append(state.upper())

        if city:
            conditions.append("city LIKE %s")
            params.append(f"%{city}%")

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

        # Get total count
        cur.execute(f"SELECT COUNT(*) as total FROM job_descriptions {where_clause}", params)
        total = cur.fetchone()["total"]

        # Paginated results
        offset = (page - 1) * page_size
        cur.execute(f"""
            SELECT job_id, job_diva_no, title, company_name, division_name,
                   position_type, job_status, city, state, zipcode, country,
                   address1, address2, issue_date, start_date, end_date,
                   max_allowed_submittals
            FROM job_descriptions
            {where_clause}
            ORDER BY issue_date DESC
            LIMIT %s OFFSET %s
        """, params + [page_size, offset])
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    # Transform to camelCase for frontend (only columns needed for the grid)
    data = []
    for row in rows:
        data.append({
            "jobId": row["job_id"],
            "jobDivaNo": row["job_diva_no"],
            "title": row["title"],
            "companyName": row["company_name"],
            "divisionName": row["division_name"],
            "jobStatus": row["job_status"],
            "positionType": row["position_type"],
            "city": row["city"],
            "state": row["state"],
            "country": row["country"],
            "issueDate": row["issue_date"].isoformat() if row["issue_date"] else None,
            "startDate": row["start_date"].isoformat() if row["start_date"] else None,
            "endDate": row["end_date"].isoformat() if row["end_date"] else None,
        })

    return {"data": data, "total": total}


@router.get("/filters")
def get_filter_options():
    """Return distinct values for all filter dropdowns — always in sync with DB."""
    conn = db_pool.get_connection()
    try:
        cur = conn.cursor(dictionary=True)

        cur.execute("SELECT DISTINCT job_status FROM job_descriptions WHERE job_status IS NOT NULL AND job_status != '' ORDER BY job_status")
        statuses = [r["job_status"] for r in cur.fetchall()]

        cur.execute("SELECT DISTINCT position_type FROM job_descriptions WHERE position_type IS NOT NULL AND position_type != '' ORDER BY position_type")
        position_types = [r["position_type"] for r in cur.fetchall()]

        cur.execute("SELECT DISTINCT TRIM(company_name) as cn FROM job_descriptions WHERE company_name IS NOT NULL AND TRIM(company_name) != '' ORDER BY cn")
        companies = sorted(set(r["cn"] for r in cur.fetchall()))

        cur.execute("SELECT DISTINCT country FROM job_descriptions WHERE country IS NOT NULL AND country != '' ORDER BY country")
        countries = [r["country"] for r in cur.fetchall()]

        cur.execute("SELECT DISTINCT state FROM job_descriptions WHERE state IS NOT NULL AND state != '' AND LENGTH(state) <= 3 ORDER BY state")
        states = [r["state"] for r in cur.fetchall()]

        cur.execute("SELECT DISTINCT city FROM job_descriptions WHERE city IS NOT NULL AND city != '' AND city REGEXP '^[A-Za-z]' ORDER BY city")
        cities = [r["city"] for r in cur.fetchall()]

        cur.close()
    finally:
        conn.close()

    return {
        "statuses": statuses,
        "positionTypes": position_types,
        "companies": companies,
        "countries": countries,
        "states": states,
        "cities": cities,
    }


@router.get("/stats")
def get_jd_stats():
    """Quick stats about the synced JD data."""
    conn = db_pool.get_connection()
    try:
        cur = conn.cursor(dictionary=True)

        cur.execute("""
            SELECT
                COUNT(*) as total_jobs,
                SUM(CASE WHEN job_status = 'OPEN' THEN 1 ELSE 0 END) as open_jobs,
                SUM(CASE WHEN job_status = 'CLOSED' THEN 1 ELSE 0 END) as closed_jobs,
                MIN(issue_date) as earliest_date,
                MAX(issue_date) as latest_date,
                COUNT(DISTINCT company_name) as companies,
                COUNT(DISTINCT city) as cities
            FROM job_descriptions
        """)
        stats = cur.fetchone()
        cur.close()
    finally:
        conn.close()

    return {
        "totalJobs": stats["total_jobs"],
        "openJobs": stats["open_jobs"] or 0,
        "closedJobs": stats["closed_jobs"] or 0,
        "earliestDate": stats["earliest_date"].isoformat() if stats["earliest_date"] else None,
        "latestDate": stats["latest_date"].isoformat() if stats["latest_date"] else None,
        "companies": stats["companies"],
        "cities": stats["cities"],
    }


@router.get("/sync-status")
def sync_status():
    """Check when the last sync happened and when next is due."""
    conn = db_pool.get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("SELECT MAX(issue_date) as latest FROM job_descriptions")
        row = cur.fetchone()
        cur.execute("SELECT COUNT(*) as cnt FROM job_descriptions WHERE issue_date >= DATE_SUB(NOW(), INTERVAL 14 DAY)")
        recent = cur.fetchone()
        cur.close()
    finally:
        conn.close()

    from datetime import datetime, timedelta
    latest = row["latest"] if row else None
    next_sync = (latest + timedelta(days=1)).isoformat() if latest else "pending"

    return {
        "autoSyncEnabled": True,
        "intervalDays": 1,
        "latestJobDate": latest.isoformat() if latest else None,
        "recentJobs14d": recent["cnt"],
        "nextSyncEstimate": next_sync,
        "manualTrigger": "/api/jds/sync?days=1",
    }


@router.get("/sync")
def trigger_sync(days: int = Query(1, description="Fetch last N days from JobDiva")):
    """Manually trigger a sync from JobDiva API to DB.
    
    This endpoint fetches fresh data from JobDiva and upserts into the DB.
    The server also auto-syncs every day via background scheduler.
    """
    try:
        token = _authenticate()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"JobDiva auth failed: {str(e)}")

    now = datetime.now()
    from_dt = now - timedelta(days=days)
    to_dt = now

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    all_records = []
    chunk_start = from_dt
    DATE_FMT = "%m/%d/%Y %H:%M:%S"

    while chunk_start < to_dt:
        chunk_end = min(chunk_start + timedelta(days=14), to_dt)
        params = {
            "fromDate": chunk_start.strftime(DATE_FMT),
            "toDate": chunk_end.strftime(DATE_FMT),
        }
        try:
            resp = requests.get(
                f"{BASE_URL}/bi/NewUpdatedJobRecords",
                params=params,
                headers=headers,
                timeout=60,
            )
            resp.raise_for_status()
            data = resp.json()
            records = data.get("data", [])
            if isinstance(records, list):
                all_records.extend(records)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"JobDiva API error: {str(e)}")
        chunk_start = chunk_end

    # Upsert to DB
    conn = db_pool.get_connection()
    try:
        cur = conn.cursor()
        upserted = 0
        for r in all_records:
            try:
                cur.execute("""
                    INSERT INTO job_descriptions (
                        job_id, job_diva_no, title, company_name, division_name,
                        position_type, job_status, city, state, zipcode, country,
                        address1, address2, issue_date, start_date, end_date,
                        max_allowed_submittals
                    ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                    ON DUPLICATE KEY UPDATE
                        job_diva_no=VALUES(job_diva_no), title=VALUES(title),
                        company_name=VALUES(company_name), division_name=VALUES(division_name),
                        position_type=VALUES(position_type), job_status=VALUES(job_status),
                        city=VALUES(city), state=VALUES(state), zipcode=VALUES(zipcode),
                        country=VALUES(country), address1=VALUES(address1), address2=VALUES(address2),
                        issue_date=VALUES(issue_date), start_date=VALUES(start_date),
                        end_date=VALUES(end_date), max_allowed_submittals=VALUES(max_allowed_submittals)
                """, (
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
                    _parse_dt(r.get("ISSUEDATE")),
                    _parse_dt(r.get("STARTDATE")),
                    _parse_dt(r.get("ENDDATE")),
                    _safe_int(r.get("MAXALLOWEDSUBMITTALS")),
                ))
                upserted += 1
            except Exception:
                pass
        conn.commit()
        cur.close()
    finally:
        conn.close()

    return {
        "message": f"Synced {upserted} job descriptions from JobDiva",
        "fetched": len(all_records),
        "upserted": upserted,
        "dateRange": f"{from_dt.strftime('%m/%d/%Y')} to {to_dt.strftime('%m/%d/%Y')}",
    }


@router.get("/interviewers/{job_id}")
def get_interviewer_panel(
    job_id: str,
    top_n: int = Query(10, ge=1, le=50, description="Max interviewers to return"),
    min_score: float = Query(0.2, ge=0.0, le=1.0, description="Minimum match score"),
):
    """
    Find the best internal employees to serve as interviewers for a JD.
    
    Scoring uses PRIMARY skill overlap (weight 1.0) + SECONDARY (weight 0.3).
    Only employees with classified skills are considered.
    """
    conn = db_pool.get_connection()
    try:
        result = find_interviewers(conn, job_id, top_n=top_n, min_score=min_score)
    finally:
        conn.close()
    
    return result


@router.get("/employee-skills/{employee_id}")
def get_employee_skills_profile(employee_id: int):
    """
    Get full skill profile for an employee with PRIMARY/SECONDARY classification.
    Used for interviewer panel drill-down.
    """
    conn = db_pool.get_connection()
    try:
        result = get_employee_skill_profile(conn, employee_id)
    finally:
        conn.close()
    
    if not result["summary"]["totalSkills"]:
        raise HTTPException(status_code=404, detail=f"No skills found for employee {employee_id}")
    
    return result


@router.get("/{job_id}")
def get_job_detail(job_id: str):
    """Get full detail of a specific job. Fetches description from JobDiva if not cached."""
    conn = db_pool.get_connection()
    try:
        cur = conn.cursor(dictionary=True)

        cur.execute("SELECT * FROM job_descriptions WHERE job_id = %s", (job_id,))
        row = cur.fetchone()

        # If we have the row but no description, fetch it from JobDiva and cache it
        if row and not row.get("description"):
            try:
                token = _authenticate()
                headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
                resp = requests.get(
                    f"{BASE_URL}/bi/JobsDetail",
                    params={"jobIds": job_id},
                    headers=headers,
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
                records = data.get("data", [])
                if records:
                    raw_description = records[0].get("JOBDESCRIPTION") or ""
                    # Clean out internal admin notes, keep only the actual JD
                    description = _clean_description(raw_description)
                    # Cache the cleaned version in DB
                    cur.execute(
                        "UPDATE job_descriptions SET description = %s WHERE job_id = %s",
                        (description, job_id)
                    )
                    conn.commit()
                    row["description"] = description
            except Exception:
                row["description"] = None

        cur.close()
    finally:
        conn.close()

    if row:
        return {
            "jobDivaNo": row["job_diva_no"],
            "title": row["title"],
            "companyName": row["company_name"],
            "divisionName": row["division_name"],
            "positionType": row["position_type"],
            "jobStatus": row["job_status"],
            "city": row["city"],
            "state": row["state"],
            "country": row["country"],
            "issueDate": row["issue_date"].isoformat() if row["issue_date"] else None,
            "startDate": row["start_date"].isoformat() if row["start_date"] else None,
            "endDate": row["end_date"].isoformat() if row["end_date"] else None,
            "description": row.get("description"),
        }

    # Fallback: fetch from JobDiva API
    try:
        token = _authenticate()
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        resp = requests.get(
            f"{BASE_URL}/bi/JobsDetail",
            params={"jobIds": job_id},
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        records = data.get("data", [])
        if not records:
            raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
        return _transform_detail(records[0])
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found in DB or JobDiva")


# ─── Helper functions ─────────────────────────────────────────────

def _authenticate() -> str:
    """Get JWT token from JobDiva."""
    client_id = os.getenv("JOBDIVA_CLIENT_ID")
    username = os.getenv("JOBDIVA_USERNAME")
    password = os.getenv("JOBDIVA_PASSWORD")

    if not all([client_id, username, password]):
        raise HTTPException(status_code=500, detail="JobDiva credentials not configured")

    url = f"{BASE_URL}/authenticate"
    params = {"clientid": client_id, "username": username, "password": password}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.text.strip()


def _parse_dt(val):
    """Parse datetime from JobDiva response."""
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


def _safe_int(val):
    """Safely convert to int."""
    try:
        return int(val) if val else 0
    except (ValueError, TypeError):
        return 0


def _clean_description(raw_desc):
    """Extract actual job description, removing header metadata and internal notes.
    
    JobDiva descriptions often contain:
    - Header metadata: Beeline Job Title, Quantity, Bill Rate, Location, etc.
    - Actual job description: The real JD content (what we want)
    - Footer internal notes: Interview process, rate guidance, conversion info, PPE, etc.
    
    Some descriptions use '______' dividers; others use structured headings.
    """
    if not raw_desc:
        return None

    import re

    # Normalize: replace HTML <br> tags with newlines for splitting
    text = raw_desc.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n")

    # ── CASE 1: Underscore dividers ──
    sections = re.split(r'_{10,}', text)
    if len(sections) > 1:
        for section in reversed(sections):
            cleaned = section.strip()
            if len(cleaned) > 100:
                cleaned = cleaned.replace("\n", "<br>")
                return cleaned
        last = sections[-1].strip()
        if last:
            return last.replace("\n", "<br>")

    # ── CASE 2: Structured header/footer format ──
    # Find the start of the actual JD (after metadata header)
    # Look for "Job Description:" as the start marker
    jd_start_pattern = re.search(
        r'(?:^|\n)\s*Job\s*Description\s*[:\-]\s*\n?',
        text, re.IGNORECASE
    )

    if jd_start_pattern:
        # Extract everything after "Job Description:"
        text = text[jd_start_pattern.end():]
    else:
        # No explicit marker — try stripping known header fields
        header_fields = [
            r'(?:Description\s*:\s*)?Beeline\s+Job\s+Title\s*:.*',
            r'Preferred\s+Job\s+Title\s*:.*',
            r'Quantity\s+Needed\s*:.*',
            r'Initial\s+Assignment\s+Length\s*:.*',
            r'Shift/?Schedule\s+Details?\s*:.*',
            r'Location\s+Specifics?\s*:.*',
            r'Hourly\s+Bill\s+Rate\s*:.*',
            r'Max\s+Bill\s+Rate\s*:.*',
            r'Target\s+Bill\s+Rate\s*:.*',
            r'Duration\s*:.*',
            r'Start\s+Date\s*:.*',
            r'End\s+Date\s*:.*',
        ]
        for field_pat in header_fields:
            text = re.sub(r'(?:^|\n)\s*' + field_pat, '', text, flags=re.IGNORECASE)

    # ── Strip footer / internal notes ──
    # Cut off at the first internal-notes section header
    footer_markers = [
        r'What\s+will\s+the\s+candidate.?s\s+interview',
        r'Is\s+there\s+any\s+possibility\s+for\s+future\s+conversion',
        r'Does\s+this\s+role\s+require\s+PPE',
        r'Will\s+there\s+be\s+a\s+supplier\s+spotlight',
        r'Rate\s+Guidance\s+by\s+Candidate',
        r'Rate\s+Guidance\s*:',
        r'Supplier\s+Engagement\s+Questions?\s*:',
        r'Reason\s+for\s+request\s*:',
        r'Work\s+Environment\s*(&|and)?\s*Team\s+Structure',
        r'Note\s*:\s*The\s+hiring\s+manager',
        r'Hourly\s+Bill\s+Rate\s*:',  # if it appears in body (shouldn't after header strip)
    ]
    for marker in footer_markers:
        match = re.search(r'(?:^|\n)\s*' + marker, text, re.IGNORECASE)
        if match:
            text = text[:match.start()]

    # Clean up
    text = text.strip()

    if not text or len(text) < 50:
        # If stripping removed everything, fall back to original
        return raw_desc.replace("<br>", "\n").replace("<br/>", "\n").replace("<br />", "\n").replace("\n", "<br>")

    # Convert back to HTML for frontend rendering
    return text.replace("\n", "<br>")


def _transform_detail(record: dict) -> dict:
    """Transform a JobsDetail record (fallback for API call)."""
    return {
        "jobDivaNo": record.get("JOBDIVANO"),
        "title": record.get("JOBTITLE"),
        "companyName": record.get("COMPANYNAME"),
        "divisionName": record.get("DIVISION"),
        "positionType": record.get("POSITIONTYPE"),
        "jobStatus": record.get("JOBSTATUS"),
        "city": record.get("CITY"),
        "state": record.get("STATE"),
        "country": record.get("COUNTRY"),
        "issueDate": record.get("DATEISSUED"),
        "startDate": record.get("STARTDATE"),
        "endDate": record.get("ENDDATE"),
        "description": _clean_description(record.get("JOBDESCRIPTION")),
    }
