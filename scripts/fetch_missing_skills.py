"""
Fetch JD descriptions from JobDiva for titles missing skills,
then extract skills and insert into jd_extracted_skills.
"""
import os, sys, json, requests, time, re
from datetime import datetime
from dotenv import load_dotenv
import mysql.connector
from openai import AzureOpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DB_CFG = dict(host='localhost', port=3305, user='root', password='rootpassword', database='resume_processing')
BASE_URL = "https://api.jobdiva.com/apiv2"

# Azure OpenAI
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
)

def authenticate():
    cid = os.getenv("JOBDIVA_CLIENT_ID")
    user = os.getenv("JOBDIVA_USERNAME")
    pw = os.getenv("JOBDIVA_PASSWORD")
    resp = requests.get(f"{BASE_URL}/authenticate", params={"clientid": cid, "username": user, "password": pw}, timeout=30)
    resp.raise_for_status()
    return resp.text.strip()

def fetch_description(token, job_id):
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    resp = requests.get(f"{BASE_URL}/bi/JobsDetail", params={"jobIds": job_id}, headers=headers, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    records = data.get("data", [])
    if records:
        return records[0].get("JOBDESCRIPTION", "")
    return ""

def clean_description(raw):
    """Remove internal admin notes."""
    if not raw:
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', raw)
    text = re.sub(r'&[a-z]+;', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_skills_llm(title, description):
    """Extract skills using Azure OpenAI."""
    prompt = f"""Extract the top 5-8 technical skills/technologies required for this job.
Return ONLY a JSON array of skill names. Be specific and technical.
Do not include soft skills, generic terms, or company names.

Job Title: {title}
Job Description: {description[:3000]}

Return format: ["Skill 1", "Skill 2", ...]"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        text = resp.choices[0].message.content.strip()
        # Parse JSON array
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"  LLM error: {e}")
    return []

def main():
    conn = mysql.connector.connect(**DB_CFG)
    cur = conn.cursor(dictionary=True)

    # Find job IDs for the missing titles
    titles = ['Telecommunications Engineer - III', 'Software Development Engineer 2']
    
    print("Authenticating with JobDiva...")
    token = authenticate()
    print("  OK")

    total_inserted = 0
    
    for title in titles:
        print(f"\n=== {title} ===")
        cur.execute("""
            SELECT job_id FROM updated_job_records
            WHERE role_cluster = 'Engineering - General' AND title = %s AND country = 'US'
              AND issue_date >= '2026-01-01' AND issue_date < '2026-04-01'
            ORDER BY issue_date DESC
            LIMIT 10
        """, (title,))
        job_ids = [r['job_id'] for r in cur.fetchall()]
        print(f"  Found {len(job_ids)} job records in Q1")

        fetched = 0
        for job_id in job_ids:
            # Check if we already have a description cached
            cur.execute("SELECT description FROM job_descriptions WHERE job_id = %s", (str(job_id),))
            row = cur.fetchone()
            
            if row and row.get('description'):
                desc = row['description']
                print(f"  job_id={job_id}: cached description ({len(desc)} chars)")
            else:
                # Fetch from JobDiva
                try:
                    raw = fetch_description(token, str(job_id))
                    desc = clean_description(raw)
                    if desc:
                        # Cache it
                        if row:
                            cur.execute("UPDATE job_descriptions SET description = %s WHERE job_id = %s", (desc, str(job_id)))
                        else:
                            cur.execute("""
                                INSERT INTO job_descriptions (job_id, title, description)
                                VALUES (%s, %s, %s)
                                ON DUPLICATE KEY UPDATE description = VALUES(description)
                            """, (str(job_id), title, desc))
                        conn.commit()
                        print(f"  job_id={job_id}: fetched from JobDiva ({len(desc)} chars)")
                        fetched += 1
                    else:
                        print(f"  job_id={job_id}: no description in JobDiva")
                        continue
                except Exception as e:
                    print(f"  job_id={job_id}: fetch error - {e}")
                    continue
                time.sleep(0.5)  # Rate limit

            if not desc or len(desc) < 50:
                continue

            # Extract skills
            skills = extract_skills_llm(title, desc)
            if skills:
                print(f"    Skills: {skills}")
                # Insert into jd_extracted_skills
                cur.execute("""
                    INSERT INTO jd_extracted_skills 
                    (job_id, cluster_name, company_name, country, title, extracted_skills, 
                     skill_count, extraction_method, source_type, quarter, extracted_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE extracted_skills = VALUES(extracted_skills)
                """, (
                    str(job_id), 'US | Engineering - General', '', 'US', title,
                    json.dumps(skills), len(skills), 'llm_gpt4o_mini', 'full_text',
                    '2026Q1', datetime.now()
                ))
                conn.commit()
                total_inserted += 1
            else:
                print(f"    No skills extracted")

            if fetched >= 5:
                break  # Don't hammer the API

    cur.close()
    conn.close()
    print(f"\n=== Done: {total_inserted} skill records inserted ===")

if __name__ == "__main__":
    main()
