

import os
import sys
import base64
import mysql.connector
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from resume_agent.extractors.openai_extractor import OpenAIResumeExtractor
from resume_agent.integrations.jobdiva_client import JobDivaClient
from resume_agent.file_storage import FileStorageConfig


class ResumeAgent:

    def __init__(self):
        if not (os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_KEY")):
            raise RuntimeError("OPENAI_API_KEY or AZURE_OPENAI_KEY not set")
        if not os.getenv("MYSQL_HOST"):
            raise RuntimeError("MYSQL env vars not set")

        self.extractor = OpenAIResumeExtractor()
        self.jobdiva = JobDivaClient()
        FileStorageConfig.initialize_directories()

        self.db_config = {
            "host": os.getenv("MYSQL_HOST"),
            "port": int(os.getenv("MYSQL_PORT")),
            "database": os.getenv("MYSQL_DATABASE"),
            "user": os.getenv("MYSQL_USER"),
            "password": os.getenv("MYSQL_PASSWORD"),
        }

        print("\n" + "=" * 60)
        print("RESUME AGENT INITIALIZED")
        print("=" * 60 + "\n")

   

    def _detect_file_type(self, base64_content):
        try:
            decoded = base64.b64decode(base64_content[:100])
            if decoded.startswith(b"%PDF"):
                return "pdf"
            if decoded.startswith(b"PK"):
                return "docx"
            if decoded.startswith(b"\xd0\xcf\x11\xe0"):
                return "doc"
            return "pdf"
        except Exception:
            return "pdf"

    def save_original_file(self, resume_id, base64_content, file_type):
        file_path = FileStorageConfig.get_resume_path(resume_id, file_type)
        saved_path = FileStorageConfig.save_base64_to_file(base64_content, file_path)
        print(f"[AGENT] Original file saved: {saved_path}")
        return saved_path


    # DB Operations (DIRECT - resume table)

    def save_base64_direct(self, resume_id, employee_jobdiva_id, base64_data, file_type, file_path):
        """Save base64 + file info directly to resume table"""
        print("[AGENT] Saving base64 to DB (DIRECT)")

        conn = mysql.connector.connect(**self.db_config)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO resume (resume_id, employee_jobdiva_id, resume_base64, file_type, file_path)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                resume_base64       = VALUES(resume_base64),
                file_type           = VALUES(file_type),
                file_path           = VALUES(file_path)
            """,
            (resume_id, employee_jobdiva_id, base64_data, file_type, file_path),
        )

        conn.commit()
        cursor.close()
        conn.close()

        print("[AGENT] Base64 saved successfully")

    def fetch_base64_from_db(self, resume_id):
        """Fetch base64 from resume table"""
        print(f"[AGENT] Fetching base64 FROM DB for {resume_id}")

        conn = mysql.connector.connect(**self.db_config)
        cursor = conn.cursor()
        cursor.execute("SELECT resume_base64 FROM resume WHERE resume_id = %s", (resume_id,))
        row = cursor.fetchone()

        cursor.close()
        conn.close()

        if not row:
            raise RuntimeError(f"Resume {resume_id} not found in DB")

        return row[0]


    # MCP (EXTRACTED DATA ONLY)

    async def save_extracted_via_mcp(self, resume_id, employee_jobdiva_id, extracted_data):
        print("[AGENT] Saving extracted data via MCP")

        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
        )

        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                await session.call_tool(
                    "save_extracted_resume",
                    {
                        "resume_id": resume_id,
                        "employee_jobdiva_id": employee_jobdiva_id,
                        "extracted_data": extracted_data,
                    },
                )

        print("[AGENT] Extracted data saved via MCP")

    # --------------------------------------------------
    # SINGLE RESUME INGESTION
    # --------------------------------------------------

    async def ingest_jobdiva_resume(self, employee_jobdiva_id, resume_id):
        print("\n" + "=" * 70)
        print("PROCESSING RESUME")
        print("=" * 70)
        print(f"Resume ID:           {resume_id}")
        print(f"Employee JobDiva ID: {employee_jobdiva_id}")
        print("=" * 70 + "\n")

        # STEP 1: Fetch from JobDiva API
        content = self.jobdiva.get_resume_content(resume_id)
        base64_data = content["base64"]
        text = content["text"]

        if not base64_data:
            raise RuntimeError("Empty base64 from JobDiva")

        # STEP 2: Detect type
        file_type = self._detect_file_type(base64_data)

        # STEP 3: Save original file to disk
        file_path = self.save_original_file(resume_id, base64_data, file_type)

        # STEP 4: Save base64 to DB (resume table)
        self.save_base64_direct(resume_id, employee_jobdiva_id, base64_data, file_type, file_path)

        # STEP 5: Extract using OpenAI
        extracted = self.extractor.extract(text)
        extracted["score"] = float(extracted.get("score", 0.0))

        # STEP 6: Save extracted data via MCP (resume + employee_skillset tables)
        await self.save_extracted_via_mcp(resume_id, employee_jobdiva_id, extracted)

        print("\n" + "=" * 70)
        print(f"SUCCESS: {resume_id}")
        print("=" * 70 + "\n")

        return extracted


    # --------------------------------------------------
    # BATCH INGESTION FROM EMPLOYEE TABLE
    # --------------------------------------------------

    async def ingest_from_employee_table(self, limit: int = 5, latest_only: bool = True):
        """
        Pick employees from DB, fetch their resume IDs from JobDiva,
        then process each resume.
        
        Args:
            limit: Number of employees to process
            latest_only: If True, only process the latest resume per employee
        """
        print("\n" + "=" * 70)
        print(f"BATCH INGESTION FROM EMPLOYEE TABLE (limit={limit}, latest_only={latest_only})")
        print("=" * 70 + "\n")

        # Get employee IDs from DB (skip already processed)
        conn = mysql.connector.connect(**self.db_config)
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT e.id FROM employee e
            LEFT JOIN resume r ON e.id = r.employee_jobdiva_id
            WHERE r.resume_id IS NULL
            LIMIT %s
            """,
            (limit,),
        )
        employees = [row[0] for row in cursor.fetchall()]
        cursor.close()
        conn.close()

        if not employees:
            print("No unprocessed employees found!")
            return {"success": 0, "failed": 0, "skipped": 0}

        print(f"Found {len(employees)} unprocessed employees")

        success = 0
        failed = 0
        skipped = 0

        for i, emp_id in enumerate(employees, 1):
            print(f"\n[{i}/{len(employees)}] Employee ID: {emp_id}")
            try:
                # Get resume IDs from JobDiva
                resume_list = self.jobdiva._get_candidate_resume_ids(str(emp_id))

                if not resume_list:
                    print(f"  No resumes found for employee {emp_id}")
                    skipped += 1
                    continue

                if latest_only:
                    # Only process the latest resume (last in the list)
                    resume_list = [resume_list[-1]]
                    print(f"  Latest resume only: {resume_list[0]['resume_id']}")

                for resume_info in resume_list:
                    resume_id = resume_info["resume_id"]
                    await self.ingest_jobdiva_resume(
                        employee_jobdiva_id=emp_id,
                        resume_id=resume_id,
                    )
                    success += 1

            except Exception as e:
                print(f"  Failed employee {emp_id}: {e}")
                failed += 1

        print("\n" + "=" * 70)
        print("BATCH COMPLETE")
        print(f"  Success: {success}")
        print(f"  Failed:  {failed}")
        print(f"  Skipped: {skipped}")
        print("=" * 70 + "\n")

        # Auto-record skill history after processing
        if success > 0:
            self._record_skill_history()

        return {"success": success, "failed": failed, "skipped": skipped}


    def _record_skill_history(self):
        """Record current skill counts into skill_ranking_history table."""
        try:
            conn = mysql.connector.connect(**self.db_config)
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                """
                SELECT skill, COUNT(DISTINCT employee_jobdiva_id) AS employee_count
                FROM employee_skillset
                GROUP BY skill
                """
            )
            rankings = cursor.fetchall()

            if rankings:
                values = [(r["skill"], r["employee_count"]) for r in rankings]
                cursor.executemany(
                    "INSERT INTO skill_ranking_history (skill, employee_count) VALUES (%s, %s)",
                    values
                )
                conn.commit()
                print(f"  Skill history recorded: {len(rankings)} skills tracked")

            cursor.close()
            conn.close()
        except Exception as e:
            print(f"  Warning: Failed to record skill history: {e}")


    async def ingest_from_jobdiva(self, from_date: str, to_date: str):
        print("\n" + "=" * 70)
        print("JOBDIVA BATCH INGESTION")
        print("=" * 70)
        print(f"Date range: {from_date} → {to_date}")
        print("=" * 70 + "\n")

        resumes = self.jobdiva.get_resumes(from_date, to_date)
        print(f"Found {len(resumes)} resume(s)")

        success = 0
        failed = 0

        for i, item in enumerate(resumes, 1):
            print(f"\n[{i}/{len(resumes)}]")
            try:
                await self.ingest_jobdiva_resume(
                    employee_jobdiva_id=int(item["candidate_id"]),
                    resume_id=item["resume_id"],
                )
                success += 1
            except Exception as e:
                print(f" Failed {item['resume_id']}: {e}")
                failed += 1

        print("\n" + "=" * 70)
        print("BATCH COMPLETE")
        print(f"Success: {success}")
        print(f"Failed:  {failed}")
        print("=" * 70 + "\n")
