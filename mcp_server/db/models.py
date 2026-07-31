"""
DB model layer – every SQL statement the MCP server executes lives here.

Tables:
  employee            →  Employee master data (uploaded from Excel)
  resume              →  Resume data (base64 + extracted fields)
  employee_skillset   →  Normalized skills per resume per employee
"""

import sys
import json
from typing import Dict, Any


class ResumeDB:

    # --------------------------------------------------
    # fetch base64 from resume table
    # --------------------------------------------------
    @staticmethod
    def fetch_resume_base64(conn, resume_id: str) -> str:
        cur = conn.cursor()
        cur.execute(
            "SELECT resume_base64 FROM resume WHERE resume_id = %s",
            (resume_id,),
        )
        row = cur.fetchone()
        cur.close()

        if not row or not row[0]:
            raise RuntimeError(f"No base64 in DB for resume_id={resume_id}")
        return row[0]

    # --------------------------------------------------
    # save base64 + file info to resume table
    # --------------------------------------------------
    @staticmethod
    def save_resume_base64(
        conn,
        resume_id: str,
        employee_jobdiva_id: int,
        base64_data: str,
        file_type: str,
        file_path: str,
    ) -> bool:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO resume
                (resume_id, employee_jobdiva_id, resume_base64, file_type, file_path)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                resume_base64       = VALUES(resume_base64),
                file_type           = VALUES(file_type),
                file_path           = VALUES(file_path)
            """,
            (resume_id, employee_jobdiva_id, base64_data, file_type, file_path),
        )
        cur.close()
        print(f"[MCP-DB] resume base64 saved – {resume_id}", file=sys.stderr)
        return True

    # --------------------------------------------------
    # save extracted data to resume table + skills to employee_skillset
    # --------------------------------------------------
    @staticmethod
    def save_extracted_resume(
        conn,
        resume_id: str,
        employee_jobdiva_id: int,
        extracted_data: Dict[str, Any],
    ) -> bool:

        cur = conn.cursor()

        # ── Update resume table with extracted fields ─────────────
        cur.execute(
            """
            UPDATE resume SET
                email          = %s,
                phone          = %s,
                education      = %s,
                experience     = %s,
                skills         = %s,
                summary        = %s,
                score          = %s
            WHERE resume_id = %s
            """,
            (
                extracted_data.get("email", ""),
                extracted_data.get("phone", ""),
                json.dumps(extracted_data.get("education", [])),
                json.dumps(extracted_data.get("experience", [])),
                json.dumps(extracted_data.get("skills", [])),
                extracted_data.get("summary", ""),
                float(extracted_data.get("score", 0.0)),
                resume_id,
            ),
        )
        print(f"[MCP-DB] resume extracted data updated – {resume_id}", file=sys.stderr)

        # ── employee_skillset (one row per skill) ────────
        skills = [str(s).strip() for s in extracted_data.get("skills", []) if str(s).strip()]
        if skills:
            # Remove duplicates while preserving order and keep the write path idempotent.
            seen = set()
            unique_skills = []
            for sk in skills:
                key = sk.lower()
                if key in seen:
                    continue
                seen.add(key)
                unique_skills.append(sk)

            # wipe old skills for this resume, then batch-insert fresh
            cur.execute(
                "DELETE FROM employee_skillset WHERE resume_id = %s AND employee_jobdiva_id = %s",
                (resume_id, employee_jobdiva_id),
            )

            # Check if skill classifications are provided
            skill_classifications = extracted_data.get("skill_classifications", [])
            classification_map = {}
            if skill_classifications:
                for sc in skill_classifications:
                    classification_map[sc["skill"].lower()] = (sc["type"], sc.get("reason", ""))

            if classification_map:
                cur.executemany(
                    """
                    INSERT INTO employee_skillset 
                        (resume_id, employee_jobdiva_id, skill, skill_type, classification_reason, classified_at)
                    VALUES (%s, %s, %s, %s, %s, NOW())
                    ON DUPLICATE KEY UPDATE
                        skill_type = VALUES(skill_type),
                        classification_reason = VALUES(classification_reason),
                        classified_at = NOW()
                    """,
                    [
                        (
                            resume_id,
                            employee_jobdiva_id,
                            sk,
                            classification_map.get(sk.lower(), ("SECONDARY", ""))[0],
                            classification_map.get(sk.lower(), ("", "Unclassified"))[1],
                        )
                        for sk in unique_skills
                    ],
                )
            else:
                cur.executemany(
                    """
                    INSERT INTO employee_skillset (resume_id, employee_jobdiva_id, skill)
                    VALUES (%s, %s, %s)
                    ON DUPLICATE KEY UPDATE skill = skill
                    """,
                    [(resume_id, employee_jobdiva_id, sk) for sk in unique_skills],
                )
            print(f"[MCP-DB] {len(unique_skills)} skills inserted – {resume_id}", file=sys.stderr)

        cur.close()
        return True