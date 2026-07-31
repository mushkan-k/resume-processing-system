"""
Skill Classification Engine
----------------------------
Classifies employee skills as PRIMARY or SECONDARY using GPT-4o analysis
of their experience history. Also builds a co-occurrence matrix from JD data
for interviewer panel matching.

Usage:
    python -m scripts.classify_skills --batch-size 10 --dry-run
    python -m scripts.classify_skills --employee-id 12345
    python -m scripts.classify_skills --all
"""
import os
import sys
import json
import time
import argparse
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

import mysql.connector
from dotenv import load_dotenv

# Load .env from project root
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(ROOT_DIR / "skill_classification.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "127.0.0.1"),
        port=int(os.getenv("MYSQL_PORT", "3305")),
        database=os.getenv("MYSQL_DATABASE", "resume_processing"),
        user=os.getenv("MYSQL_USER", "resume_user"),
        password=os.getenv("MYSQL_PASSWORD", "resume_password")
    )


def get_openai_client():
    """Initialize Azure OpenAI client."""
    from openai import AzureOpenAI
    
    api_key = os.getenv("AZURE_OPENAI_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    
    if not api_key or not endpoint:
        raise RuntimeError(
            "AZURE_OPENAI_KEY and AZURE_OPENAI_ENDPOINT must be set in .env file."
        )
    
    return AzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version
    )


CLASSIFICATION_PROMPT = """You are an expert HR analyst specializing in technical talent assessment.

Given an employee's work experience history and their extracted skill list, classify EACH skill as either PRIMARY or SECONDARY.

## Definitions:
- **PRIMARY**: A skill that is CORE to this person's professional identity. They have:
  - Used it extensively across multiple roles OR as the central focus of their career
  - Demonstrated depth (not just mentioned, but actively practiced with measurable outcomes)
  - It defines what kind of professional they are (e.g., a "Java Developer" has Java as PRIMARY)

- **SECONDARY**: A skill that the person possesses but is NOT their core competency:
  - Used tangentially, in support of primary work
  - Only used in one brief role or as a minor part of their duties
  - They know it, but wouldn't be the go-to expert in an interview panel for it

## Rules:
1. A person typically has 3-6 PRIMARY skills and the rest are SECONDARY
2. If someone has 15+ years using a skill across many roles → PRIMARY
3. If a skill appears in only 1 short role with no depth → SECONDARY
4. Management/Leadership skills are PRIMARY only if person is a dedicated manager (not IC who mentors)
5. Soft skills (communication, teamwork) are almost always SECONDARY unless the role is explicitly about them
6. Consider the SENIORITY and RECENCY of skill usage
7. When in doubt, lean SECONDARY — PRIMARY means "I would trust this person to interview others on this skill"

## Employee Experience:
{experience_text}

## Skills to Classify:
{skills_list}

## Output Format:
Return a JSON array with exactly one entry per skill:
[
  {{"skill": "skill_name", "type": "PRIMARY", "reason": "One-sentence justification"}},
  {{"skill": "skill_name", "type": "SECONDARY", "reason": "One-sentence justification"}}
]

IMPORTANT: You MUST classify every single skill listed above. Do not skip any.
Return ONLY the JSON array, no other text.
"""


def format_experience_text(experience: List[Dict]) -> str:
    """Format experience JSON into readable text for the LLM."""
    if not experience:
        return "No experience data available."

    lines = []
    for i, exp in enumerate(experience, 1):
        role = exp.get("role", "Unknown Role")
        company = exp.get("company", "Unknown Company")
        duration = exp.get("duration", "Unknown Duration")
        description = exp.get("description", "")
        
        lines.append(f"{i}. **{role}** at {company} ({duration})")
        if description:
            lines.append(f"   {description}")
        lines.append("")
    
    return "\n".join(lines)


def classify_employee_skills(
    client,
    experience: List[Dict],
    skills: List[str],
    model: str = "gpt-4o"
) -> List[Dict[str, str]]:
    """
    Use GPT-4o to classify each skill as PRIMARY or SECONDARY.
    Returns list of {skill, type, reason} dicts.
    """
    if not skills:
        return []

    experience_text = format_experience_text(experience)
    skills_list = "\n".join(f"- {s}" for s in skills)

    prompt = CLASSIFICATION_PROMPT.format(
        experience_text=experience_text,
        skills_list=skills_list
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise HR classification engine. Return only valid JSON."},
            {"role": "user", "content": prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    raw = response.choices[0].message.content.strip()
    
    # Parse the response
    try:
        parsed = json.loads(raw)
        # Handle if response is wrapped in a key
        if isinstance(parsed, dict):
            # Find the list in the dict
            for key, val in parsed.items():
                if isinstance(val, list):
                    parsed = val
                    break
            else:
                raise ValueError(f"Unexpected response structure: {list(parsed.keys())}")
        
        if not isinstance(parsed, list):
            raise ValueError(f"Expected list, got {type(parsed)}")
            
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.error(f"Raw response: {raw[:500]}")
        raise

    # Validate and normalize
    results = []
    classified_skills = set()
    
    for item in parsed:
        skill_name = item.get("skill", "").strip()
        skill_type = item.get("type", "").upper().strip()
        reason = item.get("reason", "No reason provided")
        
        if skill_type not in ("PRIMARY", "SECONDARY"):
            logger.warning(f"Invalid type '{skill_type}' for skill '{skill_name}', defaulting to SECONDARY")
            skill_type = "SECONDARY"
        
        if skill_name:
            results.append({
                "skill": skill_name,
                "type": skill_type,
                "reason": reason
            })
            classified_skills.add(skill_name.lower())
    
    # Check for missed skills
    for skill in skills:
        if skill.lower() not in classified_skills:
            logger.warning(f"Skill '{skill}' was not classified by LLM, defaulting to SECONDARY")
            results.append({
                "skill": skill,
                "type": "SECONDARY",
                "reason": "Not classified by LLM — defaulted to SECONDARY for safety"
            })
    
    return results


def get_unclassified_employees(conn, limit: Optional[int] = None) -> List[Dict]:
    """Get employees whose skills haven't been classified yet."""
    cur = conn.cursor(dictionary=True)
    
    query = """
        SELECT DISTINCT es.employee_jobdiva_id, es.resume_id
        FROM employee_skillset es
        WHERE es.skill_type IS NULL
    """
    if limit:
        query += f" LIMIT {limit}"
    
    cur.execute(query)
    employees = cur.fetchall()
    cur.close()
    return employees


def get_employee_data(conn, resume_id: str) -> Tuple[List[Dict], List[str]]:
    """Get experience and skills for an employee."""
    cur = conn.cursor(dictionary=True)
    
    # Get experience from resume table
    cur.execute("SELECT experience, skills FROM resume WHERE resume_id = %s", (resume_id,))
    row = cur.fetchone()
    
    if not row:
        cur.close()
        return [], []
    
    experience = json.loads(row["experience"]) if row["experience"] else []
    
    # Get skills from employee_skillset (authoritative source)
    cur.execute(
        "SELECT skill FROM employee_skillset WHERE resume_id = %s",
        (resume_id,)
    )
    skills = [r["skill"] for r in cur.fetchall()]
    
    cur.close()
    return experience, skills


def save_classifications(
    conn, resume_id: str, employee_id: int, classifications: List[Dict]
):
    """Update employee_skillset with classification results."""
    cur = conn.cursor()
    now = datetime.now()
    
    for item in classifications:
        cur.execute("""
            UPDATE employee_skillset
            SET skill_type = %s,
                classification_reason = %s,
                classified_at = %s
            WHERE resume_id = %s
              AND employee_jobdiva_id = %s
              AND skill = %s
        """, (
            item["type"],
            item["reason"],
            now,
            resume_id,
            employee_id,
            item["skill"]
        ))
    
    conn.commit()
    cur.close()


def process_batch(
    batch_size: int = 10,
    dry_run: bool = False,
    employee_id: Optional[int] = None
):
    """Process a batch of employees for skill classification."""
    conn = get_db_connection()
    client = get_openai_client()
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
    
    if employee_id:
        cur = conn.cursor(dictionary=True)
        cur.execute(
            "SELECT DISTINCT employee_jobdiva_id, resume_id FROM employee_skillset WHERE employee_jobdiva_id = %s",
            (employee_id,)
        )
        employees = cur.fetchall()
        cur.close()
    else:
        employees = get_unclassified_employees(conn, limit=batch_size)
    
    if not employees:
        logger.info("No unclassified employees found. All done!")
        return
    
    logger.info(f"Processing {len(employees)} employees...")
    
    success_count = 0
    error_count = 0
    
    for emp in employees:
        resume_id = emp["resume_id"]
        emp_id = emp["employee_jobdiva_id"]
        
        try:
            experience, skills = get_employee_data(conn, resume_id)
            
            if not skills:
                logger.warning(f"Employee {emp_id}: No skills found, skipping")
                continue
            
            logger.info(
                f"Employee {emp_id}: Classifying {len(skills)} skills "
                f"(experience: {len(experience)} roles)"
            )
            
            classifications = classify_employee_skills(
                client, experience, skills, model=model
            )
            
            primary_count = sum(1 for c in classifications if c["type"] == "PRIMARY")
            secondary_count = sum(1 for c in classifications if c["type"] == "SECONDARY")
            
            logger.info(
                f"  => {primary_count} PRIMARY, {secondary_count} SECONDARY"
            )
            
            if dry_run:
                for c in classifications:
                    logger.info(f"    [{c['type']}] {c['skill']}: {c['reason']}")
            else:
                save_classifications(conn, resume_id, emp_id, classifications)
                success_count += 1
            
            # Rate limit: stay under 60 RPM for GPT-4o
            time.sleep(1.5)
            
        except Exception as e:
            logger.error(f"Employee {emp_id}: Classification failed: {e}")
            error_count += 1
            continue
    
    logger.info(
        f"Batch complete: {success_count} succeeded, {error_count} failed"
    )
    conn.close()


def classify_all():
    """Classify all unclassified employees in batches."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(DISTINCT employee_jobdiva_id) FROM employee_skillset WHERE skill_type IS NULL")
    total = cur.fetchone()[0]
    cur.close()
    conn.close()
    
    logger.info(f"Total unclassified employees: {total}")
    
    batch_size = 20
    processed = 0
    
    while processed < total:
        logger.info(f"Processing batch {processed // batch_size + 1} ({processed}/{total})")
        process_batch(batch_size=batch_size)
        processed += batch_size
        
        # Check remaining
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(DISTINCT employee_jobdiva_id) FROM employee_skillset WHERE skill_type IS NULL")
        remaining = cur.fetchone()[0]
        cur.close()
        conn.close()
        
        if remaining == 0:
            break
        
        logger.info(f"Remaining: {remaining}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify employee skills as PRIMARY/SECONDARY")
    parser.add_argument("--batch-size", type=int, default=10, help="Number of employees per batch")
    parser.add_argument("--dry-run", action="store_true", help="Print classifications without saving")
    parser.add_argument("--employee-id", type=int, help="Classify a specific employee")
    parser.add_argument("--all", action="store_true", help="Process ALL unclassified employees")
    
    args = parser.parse_args()
    
    if args.all:
        classify_all()
    else:
        process_batch(
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            employee_id=args.employee_id
        )
