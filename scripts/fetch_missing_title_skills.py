"""
Fetch JD descriptions from JobDiva for specific TITLES that appear in 
job decomposition but have no extracted skills in jd_extracted_skills.
These are real openings from updated_job_records - we just lack their JD text/skills.
"""
import os, sys, json, requests, time, re
from datetime import datetime
from dotenv import load_dotenv
import mysql.connector
from openai import AzureOpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

DB_CFG = dict(host='localhost', port=3305, user='root', password='rootpassword', database='resume_processing')
BASE_URL = "https://api.jobdiva.com/apiv2"

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
)

def authenticate():
    resp = requests.get(f"{BASE_URL}/authenticate", params={
        "clientid": os.getenv("JOBDIVA_CLIENT_ID"),
        "username": os.getenv("JOBDIVA_USERNAME"),
        "password": os.getenv("JOBDIVA_PASSWORD")
    }, timeout=30)
    resp.raise_for_status()
    return resp.text.strip()

def fetch_description(token, job_id):
    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    resp = requests.get(f"{BASE_URL}/bi/JobsDetail", params={"jobIds": job_id}, headers=headers, timeout=30)
    resp.raise_for_status()
    records = resp.json().get("data", [])
    return records[0].get("JOBDESCRIPTION", "") if records else ""

def clean_description(raw):
    if not raw: return ""
    text = re.sub(r'<[^>]+>', ' ', raw)
    text = re.sub(r'&[a-z]+;', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def extract_skills_llm(title, description):
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
            temperature=0.1, max_tokens=200,
        )
        text = resp.choices[0].message.content.strip()
        match = re.search(r'\[.*?\]', text, re.DOTALL)
        if match: return json.loads(match.group())
    except Exception as e:
        print(f"    LLM error: {e}")
    return []

def main():
    conn = mysql.connector.connect(**DB_CFG)
    cur = conn.cursor(dictionary=True)

    # Find titles in Q1 decomposition that have NO extracted skills
    # These are real openings from updated_job_records
    cur.execute("""
        SELECT u.title, u.role_cluster, u.country, u.job_id, u.issue_date,
               COUNT(*) OVER (PARTITION BY u.title, u.role_cluster, u.country) as opens
        FROM updated_job_records u
        LEFT JOIN jd_extracted_skills j 
            ON j.title = u.title COLLATE utf8mb4_0900_ai_ci
            AND j.quarter = '2026Q1'
        WHERE u.issue_date >= '2026-01-01' AND u.issue_date < '2026-04-01'
            AND j.id IS NULL
        ORDER BY opens DESC, u.role_cluster, u.title
    """)
    all_rows = cur.fetchall()

    # Deduplicate: one job_id per (title, cluster, country)
    seen = set()
    targets = []
    for r in all_rows:
        key = (r['title'], r['role_cluster'], r['country'])
        if key not in seen:
            seen.add(key)
            targets.append(r)

    print(f"Unique titles missing skills: {len(targets)}")
    
    # Only process titles with >= 3 openings (impact)
    high_impact = [t for t in targets if t['opens'] >= 3]
    print(f"Titles with >= 3 openings (high impact): {len(high_impact)}")
    
    token = authenticate()
    print("JobDiva authenticated\n")
    
    inserted = 0
    skipped_no_desc = 0
    skipped_no_skills = 0

    for i, t in enumerate(high_impact):
        title = t['title']
        cluster = t['role_cluster']
        country = t['country'] or 'US'
        job_id = str(t['job_id'])
        opens = t['opens']
        cluster_name = f"{country} | {cluster}"

        print(f"[{i+1}/{len(high_impact)}] {title} ({opens} opens) | {cluster_name}")

        # Try to get description from cache first
        cur.execute("SELECT description FROM job_descriptions WHERE job_id = %s", (job_id,))
        cached = cur.fetchone()
        desc = None

        if cached and cached.get('description') and len(cached['description']) > 50:
            desc = cached['description']
        else:
            try:
                raw = fetch_description(token, job_id)
                desc = clean_description(raw)
                if desc and len(desc) > 50:
                    cur.execute("""
                        INSERT INTO job_descriptions (job_id, title, description)
                        VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE description = VALUES(description)
                    """, (job_id, title, desc))
                    conn.commit()
                else:
                    desc = None
            except Exception as e:
                print(f"  Fetch error: {e}")
            time.sleep(0.3)

        if not desc:
            skipped_no_desc += 1
            print(f"  No description available")
            continue

        # Extract skills via LLM
        skills = extract_skills_llm(title, desc)
        if not skills:
            skipped_no_skills += 1
            print(f"  LLM returned no skills")
            continue

        # Determine quarter
        issue = t['issue_date']
        if issue:
            q = f"{issue.year}Q{(issue.month-1)//3+1}"
        else:
            q = "2026Q1"

        if len(country) > 5:
            country = country[:2]

        cur.execute("""
            INSERT INTO jd_extracted_skills 
            (job_id, cluster_name, company_name, country, title, extracted_skills, 
             skill_count, extraction_method, source_type, quarter, extracted_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE extracted_skills = VALUES(extracted_skills)
        """, (
            job_id, cluster_name, '', country, title,
            json.dumps(skills), len(skills), 'llm_gpt4o_mini', 'jd_full_text',
            q, datetime.now()
        ))
        conn.commit()
        inserted += 1
        print(f"  ✓ {skills[:5]}")

    cur.close()
    conn.close()
    print(f"\n{'='*50}")
    print(f"Inserted: {inserted}")
    print(f"Skipped (no desc): {skipped_no_desc}")
    print(f"Skipped (no skills): {skipped_no_skills}")
    print(f"Total high-impact titles processed: {len(high_impact)}")

if __name__ == "__main__":
    main()
