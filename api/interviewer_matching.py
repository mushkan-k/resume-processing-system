"""
Interviewer Panel Matching Engine
---------------------------------
Finds the best internal employees to serve as interviewers for a given JD,
based on PRIMARY skill overlap with the JD's required skills.

Scoring Algorithm:
  1. Extract skills from the JD (from jd_extracted_skills table)
  2. For each employee with classified skills:
     - PRIMARY skill match → weight 1.0
     - SECONDARY skill match → weight 0.3
  3. Score = sum(weights) / total_jd_skills
  4. Rank by score descending, return top N candidates
"""
import json
import logging
from typing import List, Dict, Optional, Tuple
from collections import defaultdict

import mysql.connector

logger = logging.getLogger(__name__)


def get_jd_skills(conn, job_id: str) -> List[str]:
    """Get extracted skills for a JD from jd_extracted_skills table."""
    cur = conn.cursor(dictionary=True)
    cur.execute(
        "SELECT extracted_skills FROM jd_extracted_skills WHERE job_id = %s",
        (job_id,)
    )
    row = cur.fetchone()
    cur.close()
    
    if not row:
        return []
    
    skills_raw = row["extracted_skills"]
    if isinstance(skills_raw, str):
        skills = json.loads(skills_raw)
    else:
        skills = skills_raw
    
    # Normalize to lowercase for matching
    return [s.strip() for s in skills if s.strip()]


def get_jd_skills_from_description(conn, job_id: str) -> Dict:
    """Get JD metadata + skills."""
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT jes.extracted_skills, jes.cluster_name, jes.title,
               jd.company_name, jd.description
        FROM jd_extracted_skills jes
        LEFT JOIN job_descriptions jd ON jes.job_id = jd.job_id
        WHERE jes.job_id = %s
    """, (job_id,))
    row = cur.fetchone()
    cur.close()
    
    if not row:
        return {"skills": [], "cluster": None, "title": None}
    
    skills_raw = row["extracted_skills"]
    if isinstance(skills_raw, str):
        skills = json.loads(skills_raw)
    else:
        skills = skills_raw
    
    return {
        "skills": [s.strip() for s in skills if s.strip()],
        "cluster": row.get("cluster_name"),
        "title": row.get("title"),
        "company": row.get("company_name"),
        "description": row.get("description")
    }


def find_interviewers(
    conn,
    job_id: str,
    top_n: int = 10,
    min_score: float = 0.2
) -> Dict:
    """
    Find the best interviewer candidates for a given JD.
    
    Returns:
        {
            "jd": { job metadata },
            "interviewers": [
                {
                    "employeeId": int,
                    "employeeName": str,
                    "score": float,
                    "primaryMatches": [...],
                    "secondaryMatches": [...],
                    "totalSkills": int,
                    "clientName": str,
                    "deliveryCenter": str
                }
            ],
            "jdSkills": [...],
            "totalCandidatesScored": int
        }
    """
    # Step 1: Get JD skills
    jd_info = get_jd_skills_from_description(conn, job_id)
    jd_skills = jd_info["skills"]
    
    if not jd_skills:
        return {
            "jd": jd_info,
            "interviewers": [],
            "jdSkills": [],
            "totalCandidatesScored": 0,
            "message": "No extracted skills found for this JD"
        }
    
    # Normalize JD skills for case-insensitive matching
    jd_skills_lower = {s.lower() for s in jd_skills}
    
    # Step 2: Get all classified employees with their skills
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT 
            es.employee_jobdiva_id,
            es.skill,
            es.skill_type,
            e.employee_name,
            e.client_name,
            e.delivery_center
        FROM employee_skillset es
        JOIN employee e ON es.employee_jobdiva_id = e.id
        WHERE es.skill_type IS NOT NULL
    """)
    rows = cur.fetchall()
    cur.close()
    
    if not rows:
        return {
            "jd": jd_info,
            "interviewers": [],
            "jdSkills": jd_skills,
            "totalCandidatesScored": 0,
            "message": "No classified employees found. Run classify_skills first."
        }
    
    # Step 3: Group by employee and score
    employees = defaultdict(lambda: {
        "name": "",
        "client": "",
        "delivery_center": "",
        "primary_skills": [],
        "secondary_skills": [],
        "primary_matches": [],
        "secondary_matches": []
    })
    
    for row in rows:
        emp_id = row["employee_jobdiva_id"]
        skill = row["skill"]
        skill_type = row["skill_type"]
        
        emp = employees[emp_id]
        emp["name"] = row["employee_name"]
        emp["client"] = row["client_name"] or ""
        emp["delivery_center"] = row["delivery_center"] or ""
        
        if skill_type == "PRIMARY":
            emp["primary_skills"].append(skill)
            if skill.lower() in jd_skills_lower:
                emp["primary_matches"].append(skill)
        else:
            emp["secondary_skills"].append(skill)
            if skill.lower() in jd_skills_lower:
                emp["secondary_matches"].append(skill)
    
    # Step 4: Calculate scores
    PRIMARY_WEIGHT = 1.0
    SECONDARY_WEIGHT = 0.3
    total_jd_skills = len(jd_skills)
    
    scored = []
    for emp_id, emp in employees.items():
        primary_score = len(emp["primary_matches"]) * PRIMARY_WEIGHT
        secondary_score = len(emp["secondary_matches"]) * SECONDARY_WEIGHT
        raw_score = primary_score + secondary_score
        normalized_score = raw_score / total_jd_skills if total_jd_skills > 0 else 0
        
        if normalized_score >= min_score:
            scored.append({
                "employeeId": emp_id,
                "employeeName": emp["name"],
                "score": round(normalized_score, 3),
                "primaryMatches": emp["primary_matches"],
                "secondaryMatches": emp["secondary_matches"],
                "totalPrimarySkills": len(emp["primary_skills"]),
                "totalSecondarySkills": len(emp["secondary_skills"]),
                "clientName": emp["client"],
                "deliveryCenter": emp["delivery_center"]
            })
    
    # Sort by score descending
    scored.sort(key=lambda x: x["score"], reverse=True)
    
    return {
        "jd": {
            "title": jd_info.get("title"),
            "cluster": jd_info.get("cluster"),
            "company": jd_info.get("company")
        },
        "interviewers": scored[:top_n],
        "jdSkills": jd_skills,
        "totalCandidatesScored": len(scored)
    }


def get_employee_skill_profile(conn, employee_id: int) -> Dict:
    """
    Get full skill profile for an employee with PRIMARY/SECONDARY breakdown.
    Used in employee detail views and interviewer panel drill-down.
    """
    cur = conn.cursor(dictionary=True)
    cur.execute("""
        SELECT skill, skill_type, classification_reason, classified_at
        FROM employee_skillset
        WHERE employee_jobdiva_id = %s
        ORDER BY 
            CASE WHEN skill_type = 'PRIMARY' THEN 0 ELSE 1 END,
            skill
    """, (employee_id,))
    rows = cur.fetchall()
    cur.close()
    
    primary = []
    secondary = []
    unclassified = []
    
    for row in rows:
        entry = {
            "skill": row["skill"],
            "reason": row["classification_reason"],
            "classifiedAt": row["classified_at"].isoformat() if row["classified_at"] else None
        }
        if row["skill_type"] == "PRIMARY":
            primary.append(entry)
        elif row["skill_type"] == "SECONDARY":
            secondary.append(entry)
        else:
            unclassified.append(entry)
    
    return {
        "employeeId": employee_id,
        "primary": primary,
        "secondary": secondary,
        "unclassified": unclassified,
        "summary": {
            "totalSkills": len(rows),
            "primaryCount": len(primary),
            "secondaryCount": len(secondary),
            "unclassifiedCount": len(unclassified),
            "classificationComplete": len(unclassified) == 0
        }
    }
