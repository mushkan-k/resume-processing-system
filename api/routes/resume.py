"""Resume API routes"""
from fastapi import APIRouter, Query, Depends, HTTPException
from api.database import get_db
import json
import re
from datetime import datetime

router = APIRouter(prefix="/api/resumes", tags=["Resumes"])


def _parse_year(text):
    """Extract a year from text like 'April 2020', '2009', 'Jan 2016', 'Present'."""
    text = text.strip()
    if re.search(r'\b(present|current|now|ongoing|today|till\s*date|fecha|actualidad|date)\b', text, re.IGNORECASE):
        return datetime.now().year
    # Try to find a 4-digit year
    match = re.search(r'\b(19|20)\d{2}\b', text)
    if match:
        return int(match.group())
    return None


def _calculate_experience_years(exp_list):
    """
    Calculate total years of experience from a list of experience entries.
    Each entry has a 'duration' like 'April 2020 - Present' or '2009 - 2012'.
    Uses date ranges to avoid double-counting overlapping roles.
    """
    if not exp_list or not isinstance(exp_list, list):
        return None

    # Collect all (start_year, end_year) pairs
    ranges = []
    for entry in exp_list:
        duration = entry.get("duration", "") or ""
        # Normalize encoding artifacts: û → -
        duration = duration.replace('\u00fb', '-').replace('\u2013', '-').replace('\u2014', '-')
        # Split on common separators: " - ", " to ", " – ", "–", "-"
        parts = re.split(r'\s*[-–]\s*|\s+to\s+', duration, maxsplit=1)
        if len(parts) == 2:
            start_year = _parse_year(parts[0])
            end_year = _parse_year(parts[1])
            if start_year and end_year and start_year <= end_year:
                ranges.append((start_year, end_year))
        elif len(parts) == 1:
            # Single year like "2020"
            year = _parse_year(parts[0])
            if year:
                ranges.append((year, year))

    if not ranges:
        return None

    # Merge overlapping ranges to avoid double-counting
    ranges.sort()
    merged = [ranges[0]]
    for start, end in ranges[1:]:
        if start <= merged[-1][1] + 1:  # overlapping or adjacent
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Sum up total years
    total = sum(end - start for start, end in merged)
    return max(total, 1) if total > 0 else None


@router.get("")
def get_resumes(
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    search: str = Query(None),
    sort_by: str = Query("score"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    skill_filter: str = Query(None, description="Comma-separated skills to filter"),
    min_experience: int = Query(None, description="Minimum years of experience"),
    db=Depends(get_db)
):
    """
    Main endpoint for AG Grid — JOINs employee + resume + skills
    Returns data matching the Resume Catalog grid layout
    """
    cur = db.cursor(dictionary=True)

    allowed_sort = {
        "employee_id": "e.id",
        "employee_name": "e.employee_name",
        "resume_id": "r.resume_id",
        "score": "r.score",
        "email": "r.email",
        "file_type": "r.file_type",
        "created_at": "r.created_at"
    }
    order_col = allowed_sort.get(sort_by, "r.score")

    # Build WHERE clause
    conditions = []
    params = []

    if search:
        conditions.append("""
            (e.employee_name LIKE %s 
            OR r.email LIKE %s 
            OR r.resume_id LIKE %s
            OR CAST(e.id AS CHAR) LIKE %s)
        """)
        search_param = f"%{search}%"
        params.extend([search_param] * 4)

    if skill_filter:
        skills = [s.strip() for s in skill_filter.split(",")]
        placeholders = ",".join(["%s"] * len(skills))
        conditions.append(f"""
            r.resume_id IN (
                SELECT es.resume_id FROM employee_skillset es 
                WHERE es.skill IN ({placeholders})
            )
        """)
        params.extend(skills)

    where_clause = ""
    if conditions:
        where_clause = "WHERE " + " AND ".join(conditions)

    # Total count
    cur.execute(
        f"""
        SELECT COUNT(*) as total 
        FROM resume r 
        JOIN employee e ON r.employee_jobdiva_id = e.id
        {where_clause}
        """,
        params
    )
    total = cur.fetchone()["total"]

    # Paginated data (NO base64 — too large for grid)
    offset = (page - 1) * page_size
    cur.execute(
        f"""
        SELECT 
            e.id AS employee_id,
            e.employee_name,
            r.resume_id,
            r.file_type,
            r.email,
            r.phone,
            r.education,
            r.experience,
            r.skills,
            r.summary,
            r.score,
            r.created_at
        FROM resume r
        JOIN employee e ON r.employee_jobdiva_id = e.id
        {where_clause}
        ORDER BY {order_col} {sort_order}
        LIMIT %s OFFSET %s
        """,
        params + [page_size, offset]
    )
    resumes = cur.fetchall()

    # Parse JSON fields + get skills from skillset table
    resume_ids = [r["resume_id"] for r in resumes]

    # Batch fetch all skills for these resumes
    skills_map = {}
    if resume_ids:
        placeholders = ",".join(["%s"] * len(resume_ids))
        cur.execute(
            f"""
            SELECT resume_id, skill 
            FROM employee_skillset 
            WHERE resume_id IN ({placeholders})
            ORDER BY resume_id, skill
            """,
            resume_ids
        )
        for row in cur.fetchall():
            rid = row["resume_id"]
            if rid not in skills_map:
                skills_map[rid] = []
            skills_map[rid].append(row["skill"])

    # Format response for AG Grid
    for resume in resumes:
        # Parse education JSON → readable string
        if resume["education"] and isinstance(resume["education"], str):
            try:
                edu_list = json.loads(resume["education"])
                if edu_list and isinstance(edu_list, list) and len(edu_list) > 0:
                    first_edu = edu_list[0]
                    degree = first_edu.get("degree", "")
                    field = first_edu.get("field", first_edu.get("major", ""))
                    institution = first_edu.get("institution", first_edu.get("school", ""))
                    year = first_edu.get("year", first_edu.get("graduation_year", ""))
                    resume["education_display"] = f"{degree} {field} – {institution} ({year})".strip(" –")
                    resume["education"] = edu_list
                else:
                    resume["education_display"] = ""
                    resume["education"] = edu_list
            except json.JSONDecodeError:
                resume["education_display"] = str(resume["education"])
        else:
            resume["education_display"] = ""

        # Parse experience JSON → years string
        if resume["experience"] and isinstance(resume["experience"], str):
            try:
                exp_list = json.loads(resume["experience"])
                resume["experience"] = exp_list
                years = _calculate_experience_years(exp_list)
                resume["experience_years"] = f"{years} yrs" if years else ""
            except json.JSONDecodeError:
                resume["experience_years"] = ""
        elif isinstance(resume["experience"], list):
            years = _calculate_experience_years(resume["experience"])
            resume["experience_years"] = f"{years} yrs" if years else ""
        else:
            resume["experience_years"] = ""

        # Skills from skillset table (for badge display)
        resume["skills_list"] = skills_map.get(resume["resume_id"], [])

        # Parse skills JSON field too
        if resume["skills"] and isinstance(resume["skills"], str):
            try:
                resume["skills"] = json.loads(resume["skills"])
            except json.JSONDecodeError:
                pass

    cur.close()

    return {
        "data": resumes,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size
    }


@router.get("/skills")
def get_unique_skills(db=Depends(get_db)):
    """Get all unique skills with count — for filter dropdowns"""
    cur = db.cursor(dictionary=True)
    cur.execute(
        """
        SELECT skill, COUNT(*) as count 
        FROM employee_skillset 
        GROUP BY skill 
        ORDER BY count DESC
        """
    )
    skills = cur.fetchall()
    cur.close()

    return {"skills": skills, "total": len(skills)}


@router.get("/{resume_id}")
def get_resume(resume_id: str, db=Depends(get_db)):
    """Get a single resume detail (everything except base64)"""
    cur = db.cursor(dictionary=True)
    cur.execute(
        """
        SELECT 
            e.id AS employee_id,
            e.employee_name,
            e.job_diva_no,
            e.delivery_center,
            e.division_name,
            e.client_name,
            r.resume_id,
            r.file_type,
            r.file_path,
            r.email,
            r.phone,
            r.education,
            r.experience,
            r.skills,
            r.summary,
            r.score,
            r.created_at,
            r.updated_at
        FROM resume r
        JOIN employee e ON r.employee_jobdiva_id = e.id
        WHERE r.resume_id = %s
        """,
        (resume_id,)
    )
    resume = cur.fetchone()

    if not resume:
        cur.close()
        raise HTTPException(status_code=404, detail="Resume not found")

    # Parse JSON fields
    for field in ["education", "experience", "skills"]:
        if resume[field] and isinstance(resume[field], str):
            try:
                resume[field] = json.loads(resume[field])
            except json.JSONDecodeError:
                pass

    # Get skills from skillset table with classification
    cur.execute(
        "SELECT skill, skill_type, classification_reason FROM employee_skillset WHERE resume_id = %s ORDER BY CASE WHEN skill_type = 'PRIMARY' THEN 0 WHEN skill_type = 'SECONDARY' THEN 1 ELSE 2 END, skill",
        (resume_id,)
    )
    skill_rows = cur.fetchall()
    resume["skills_list"] = [row["skill"] for row in skill_rows]
    resume["skills_classified"] = [
        {
            "skill": row["skill"],
            "type": row["skill_type"],
            "reason": row["classification_reason"]
        }
        for row in skill_rows
    ]
    cur.close()

    return resume


@router.get("/{resume_id}/download")
def download_resume(resume_id: str, db=Depends(get_db)):
    """Download resume file as base64 — used by the dashboard download button"""
    cur = db.cursor(dictionary=True)
    cur.execute(
        """
        SELECT resume_id, resume_base64, file_type
        FROM resume
        WHERE resume_id = %s
        """,
        (resume_id,)
    )
    resume = cur.fetchone()
    cur.close()

    if not resume:
        raise HTTPException(status_code=404, detail="Resume not found")

    if not resume["resume_base64"]:
        raise HTTPException(status_code=404, detail="No file content available for this resume")

    file_type = resume["file_type"] or "pdf"
    file_name = f"{resume['resume_id']}.{file_type}"

    return {
        "file_name": file_name,
        "file_type": file_type,
        "base64": resume["resume_base64"]
    }