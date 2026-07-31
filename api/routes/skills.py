"""Skill Ranking API routes"""
from fastapi import APIRouter, Query, Depends, HTTPException
from api.database import get_db

router = APIRouter(prefix="/api/skills", tags=["Skills"])


@router.get("/rankings")
def get_skill_rankings(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    search: str = Query(None, description="Search skill name"),
    sort_by: str = Query("employee_count", pattern="^(skill|employee_count|primary_count|secondary_count)$"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    db=Depends(get_db)
):
    """
    Skill rankings — aggregated from employee_skillset table.
    Shows each skill + how many employees have it.
    """
    cur = db.cursor(dictionary=True)

    where_clause = ""
    params = []
    if search:
        where_clause = "WHERE es.skill LIKE %s"
        params.append(f"%{search}%")

    order_map = {
        "employee_count": "employee_count",
        "skill": "skill",
        "primary_count": "primary_count",
        "secondary_count": "secondary_count",
    }
    order_col = order_map.get(sort_by, "employee_count")

    # Total distinct skills
    cur.execute(
        f"SELECT COUNT(DISTINCT es.skill) AS total FROM employee_skillset es {where_clause}",
        params
    )
    total = cur.fetchone()["total"]

    # Paginated rankings
    offset = (page - 1) * page_size
    cur.execute(
        f"""
        SELECT
            es.skill,
            COUNT(DISTINCT es.employee_jobdiva_id) AS employee_count,
            SUM(CASE WHEN es.skill_type = 'PRIMARY' THEN 1 ELSE 0 END) AS primary_count,
            SUM(CASE WHEN es.skill_type = 'SECONDARY' THEN 1 ELSE 0 END) AS secondary_count
        FROM employee_skillset es
        {where_clause}
        GROUP BY es.skill
        ORDER BY {order_col} {sort_order}
        LIMIT %s OFFSET %s
        """,
        params + [page_size, offset]
    )
    rankings = cur.fetchall()
    cur.close()

    return {
        "data": rankings,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size
    }


@router.get("/rankings/{skill}/employees")
def get_employees_by_skill(
    skill: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    skill_type: str = Query(None, description="Filter by PRIMARY or SECONDARY"),
    db=Depends(get_db)
):
    """
    Get all employees who have a specific skill.
    Click a skill in rankings -> modal shows this list.
    Supports filtering by skill_type (PRIMARY/SECONDARY).
    """
    cur = db.cursor(dictionary=True)

    # Get type counts (always, regardless of filter)
    cur.execute(
        """SELECT
            SUM(CASE WHEN es.skill_type = 'PRIMARY' THEN 1 ELSE 0 END) AS primary_count,
            SUM(CASE WHEN es.skill_type = 'SECONDARY' THEN 1 ELSE 0 END) AS secondary_count
        FROM employee_skillset es WHERE es.skill = %s""",
        (skill,)
    )
    counts = cur.fetchone()
    primary_count = int(counts["primary_count"] or 0)
    secondary_count = int(counts["secondary_count"] or 0)

    # Build WHERE clause with optional type filter
    where_parts = ["es.skill = %s"]
    params = [skill]
    if skill_type and skill_type.upper() in ("PRIMARY", "SECONDARY"):
        where_parts.append("es.skill_type = %s")
        params.append(skill_type.upper())

    where_clause = " AND ".join(where_parts)

    cur.execute(
        f"SELECT COUNT(DISTINCT es.employee_jobdiva_id) AS total FROM employee_skillset es WHERE {where_clause}",
        params
    )
    total = cur.fetchone()["total"]

    if total == 0 and not skill_type:
        cur.close()
        raise HTTPException(status_code=404, detail=f"No employees found with skill: {skill}")

    offset = (page - 1) * page_size
    cur.execute(
        f"""
        SELECT
            e.id AS employee_id,
            e.employee_name,
            e.job_diva_no,
            e.division_name,
            e.client_name,
            es.resume_id,
            es.skill_type,
            r.score,
            r.email
        FROM employee_skillset es
        JOIN employee e ON es.employee_jobdiva_id = e.id
        LEFT JOIN resume r ON es.resume_id = r.resume_id
        WHERE {where_clause}
        GROUP BY e.id, es.resume_id, e.employee_name, e.job_diva_no,
                 e.division_name, e.client_name, es.skill_type, r.score, r.email
        ORDER BY CASE WHEN es.skill_type = 'PRIMARY' THEN 0 ELSE 1 END, e.employee_name ASC
        LIMIT %s OFFSET %s
        """,
        params + [page_size, offset]
    )
    employees = cur.fetchall()
    cur.close()

    return {
        "skill": skill,
        "data": employees,
        "total": total,
        "primary_count": primary_count,
        "secondary_count": secondary_count,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size
    }


@router.get("/history/{skill}")
def get_skill_history(
    skill: str,
    db=Depends(get_db)
):
    """
    Get the history timeline for a specific skill.
    Shows how the employee count changed over time.
    Click a skill -> modal shows this timeline.
    """
    cur = db.cursor(dictionary=True)

    cur.execute(
        """
        SELECT id, skill, employee_count, recorded_at
        FROM skill_ranking_history
        WHERE skill = %s
        ORDER BY recorded_at ASC
        """,
        (skill,)
    )
    history = cur.fetchall()

    # Also get current live count for comparison
    cur.execute(
        "SELECT COUNT(DISTINCT employee_jobdiva_id) AS current_count FROM employee_skillset WHERE skill = %s",
        (skill,)
    )
    current = cur.fetchone()["current_count"]
    cur.close()

    return {
        "skill": skill,
        "current_count": current,
        "history": history,
        "total_records": len(history)
    }


@router.post("/history/record")
def record_skill_history(db=Depends(get_db)):
    """
    Record the current employee count for ALL skills into history.
    Called automatically after the pipeline processes resumes.
    Can also be triggered manually from dashboard.
    """
    cur = db.cursor(dictionary=True)

    # Get current counts for every skill
    cur.execute(
        """
        SELECT
            es.skill,
            COUNT(DISTINCT es.employee_jobdiva_id) AS employee_count
        FROM employee_skillset es
        GROUP BY es.skill
        """
    )
    rankings = cur.fetchall()

    if not rankings:
        cur.close()
        return {"message": "No skills to record", "recorded": 0}

    # Batch insert all skills into history
    values = [(r["skill"], r["employee_count"]) for r in rankings]
    cur.executemany(
        "INSERT INTO skill_ranking_history (skill, employee_count) VALUES (%s, %s)",
        values
    )
    db.commit()
    recorded = cur.rowcount
    cur.close()

    return {
        "message": "History recorded",
        "recorded": recorded,
        "skills_tracked": len(rankings)
    }


# ==============================================================
# PRACTICE RANKINGS
# ==============================================================

@router.get("/practices")
def get_practice_rankings(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    search: str = Query(None, description="Search practice name"),
    sort_by: str = Query("employee_count", pattern="^(practice|employee_count|skill_count|primary_practitioners)$"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    db=Depends(get_db)
):
    """
    Practice rankings — aggregated from skill_practice_mapping + employee_skillset.
    Shows each practice + how many employees + how many skills + primary practitioners.
    """
    cur = db.cursor(dictionary=True)

    where_clause = ""
    params = []
    if search:
        where_clause = "WHERE spm.practice LIKE %s"
        params.append(f"%{search}%")

    order_map = {
        "practice": "practice",
        "employee_count": "employee_count",
        "skill_count": "skill_count",
        "primary_practitioners": "primary_practitioners",
    }
    order_col = order_map.get(sort_by, "employee_count")

    # Total distinct practices
    cur.execute(
        f"""
        SELECT COUNT(DISTINCT spm.practice) AS total
        FROM skill_practice_mapping spm
        {where_clause}
        """,
        params
    )
    total = cur.fetchone()["total"]

    # Paginated practice rankings with primary_practitioners count
    offset = (page - 1) * page_size
    cur.execute(
        f"""
        SELECT
            spm.practice,
            COUNT(DISTINCT es.employee_jobdiva_id) AS employee_count,
            COUNT(DISTINCT spm.skill) AS skill_count,
            COUNT(DISTINCT CASE WHEN es.skill_type = 'PRIMARY' THEN es.employee_jobdiva_id END) AS primary_practitioners
        FROM skill_practice_mapping spm
        LEFT JOIN employee_skillset es ON spm.skill = es.skill
        {where_clause}
        GROUP BY spm.practice
        ORDER BY {order_col} {sort_order}
        LIMIT %s OFFSET %s
        """,
        params + [page_size, offset]
    )
    rankings = cur.fetchall()
    cur.close()

    return {
        "data": rankings,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size
    }


@router.get("/practices/skills")
def get_skills_by_practice(
    practice: str = Query(..., description="Practice name"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    search: str = Query(None),
    sort_by: str = Query("employee_count", pattern="^(skill|employee_count)$"),
    sort_order: str = Query("desc", pattern="^(asc|desc)$"),
    db=Depends(get_db)
):
    """
    Get all skills under a specific practice with their employee counts.
    Click a practice -> see skill breakdown.
    Uses query param ?practice=... to avoid URL issues with / in names like DevOps / SRE.
    """
    cur = db.cursor(dictionary=True)

    where_parts = ["spm.practice = %s"]
    params = [practice]

    if search:
        where_parts.append("spm.skill LIKE %s")
        params.append(f"%{search}%")

    where_clause = "WHERE " + " AND ".join(where_parts)
    order_col = "employee_count" if sort_by == "employee_count" else "spm.skill"

    cur.execute(
        f"SELECT COUNT(*) AS total FROM skill_practice_mapping spm {where_clause}",
        params
    )
    total = cur.fetchone()["total"]

    if total == 0:
        cur.close()
        raise HTTPException(status_code=404, detail=f"No skills found for practice: {practice}")

    offset = (page - 1) * page_size
    cur.execute(
        f"""
        SELECT
            spm.skill,
            COUNT(DISTINCT es.employee_jobdiva_id) AS employee_count
        FROM skill_practice_mapping spm
        LEFT JOIN employee_skillset es ON spm.skill = es.skill
        {where_clause}
        GROUP BY spm.skill
        ORDER BY {order_col} {sort_order}
        LIMIT %s OFFSET %s
        """,
        params + [page_size, offset]
    )
    skills = cur.fetchall()
    cur.close()

    return {
        "practice": practice,
        "data": skills,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size
    }


@router.get("/practices/employees")
def get_employees_by_practice(
    practice: str = Query(..., description="Practice name"),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    db=Depends(get_db)
):
    """
    Get all employees who have skills in a specific practice.
    Uses query param ?practice=... to avoid URL issues with / in names like DevOps / SRE.
    """
    cur = db.cursor(dictionary=True)

    cur.execute(
        """
        SELECT COUNT(DISTINCT es.employee_jobdiva_id) AS total
        FROM employee_skillset es
        JOIN skill_practice_mapping spm ON es.skill = spm.skill
        WHERE spm.practice = %s
        """,
        (practice,)
    )
    total = cur.fetchone()["total"]

    if total == 0:
        cur.close()
        raise HTTPException(status_code=404, detail=f"No employees found for practice: {practice}")

    offset = (page - 1) * page_size
    cur.execute(
        """
        SELECT
            e.id AS employee_id,
            e.employee_name,
            e.job_diva_no,
            e.division_name,
            e.client_name,
            r.score,
            r.email,
            COUNT(DISTINCT es.skill) AS skill_count
        FROM employee_skillset es
        JOIN skill_practice_mapping spm ON es.skill = spm.skill
        JOIN employee e ON es.employee_jobdiva_id = e.id
        LEFT JOIN resume r ON es.resume_id = r.resume_id
        WHERE spm.practice = %s
        GROUP BY e.id, e.employee_name, e.job_diva_no,
                 e.division_name, e.client_name, r.score, r.email
        ORDER BY skill_count DESC, e.employee_name ASC
        LIMIT %s OFFSET %s
        """,
        (practice, page_size, offset)
    )
    employees = cur.fetchall()
    cur.close()

    return {
        "practice": practice,
        "data": employees,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": (total + page_size - 1) // page_size
    }


# ─────────────────────────────────────────────────────────────
# PRACTICE OVERLAP — chord diagram data
# ─────────────────────────────────────────────────────────────

@router.get("/practices/overlap")
def get_practice_overlap(
    min_shared: int = Query(10, ge=1, description="Minimum shared employees to include"),
    db=Depends(get_db)
):
    """
    Practice overlap data for chord diagram visualization.
    Returns pairwise employee counts between practices.
    """
    cur = db.cursor(dictionary=True)

    # Get all practices with employee counts
    cur.execute("""
        SELECT spm.practice, COUNT(DISTINCT es.employee_jobdiva_id) as employee_count
        FROM employee_skillset es
        JOIN skill_practice_mapping spm ON es.skill = spm.skill
        GROUP BY spm.practice
        ORDER BY employee_count DESC
    """)
    practices = cur.fetchall()

    # Get pairwise overlaps
    cur.execute("""
        SELECT p1.practice as source, p2.practice as target,
               COUNT(DISTINCT p1.emp) as shared_employees
        FROM (
            SELECT es.employee_jobdiva_id as emp, spm.practice
            FROM employee_skillset es
            JOIN skill_practice_mapping spm ON es.skill = spm.skill
            GROUP BY es.employee_jobdiva_id, spm.practice
        ) p1
        JOIN (
            SELECT es.employee_jobdiva_id as emp, spm.practice
            FROM employee_skillset es
            JOIN skill_practice_mapping spm ON es.skill = spm.skill
            GROUP BY es.employee_jobdiva_id, spm.practice
        ) p2 ON p1.emp = p2.emp AND p1.practice < p2.practice
        GROUP BY p1.practice, p2.practice
        HAVING shared_employees >= %s
        ORDER BY shared_employees DESC
    """, (min_shared,))
    overlaps = cur.fetchall()
    cur.close()

    # Build matrix format for chord diagram
    practice_names = [p["practice"] for p in practices]
    n = len(practice_names)
    matrix = [[0] * n for _ in range(n)]
    idx = {name: i for i, name in enumerate(practice_names)}

    for o in overlaps:
        if o["source"] in idx and o["target"] in idx:
            i, j = idx[o["source"]], idx[o["target"]]
            matrix[i][j] = o["shared_employees"]
            matrix[j][i] = o["shared_employees"]

    return {
        "practices": practices,
        "overlaps": overlaps,
        "matrix": matrix,
        "labels": practice_names
    }
