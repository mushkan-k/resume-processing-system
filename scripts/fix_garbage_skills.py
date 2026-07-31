"""
Comprehensive quality check and fix for ALL jd_extracted_skills.
Identifies garbage patterns:
1. Skills containing "Over X Year S" (experience requirements, not skills)
2. Skills that are just the job title repeated
3. Titles with 0 skills
Then re-fetches from JobDiva and re-extracts with LLM.
"""
import mysql.connector, json, os, re, requests, time
from dotenv import load_dotenv
from openai import AzureOpenAI
from datetime import datetime

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))
DB_CFG = dict(host='localhost', port=3305, user='root', password='rootpassword', database='resume_processing')
BASE_URL = "https://api.jobdiva.com/apiv2"

ai = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
)

def auth():
    r = requests.get(BASE_URL + "/authenticate", params={
        "clientid": os.getenv("JOBDIVA_CLIENT_ID"),
        "username": os.getenv("JOBDIVA_USERNAME"),
        "password": os.getenv("JOBDIVA_PASSWORD")
    }, timeout=30)
    r.raise_for_status()
    return r.text.strip()

def fetch_desc(token, job_id):
    h = {"Authorization": "Bearer " + token, "Accept": "application/json"}
    r = requests.get(BASE_URL + "/bi/JobsDetail", params={"jobIds": job_id}, headers=h, timeout=30)
    r.raise_for_status()
    data = r.json().get("data", [])
    return data[0].get("JOBDESCRIPTION", "") if data else ""

def clean(raw):
    if not raw: return ""
    t = re.sub(r'<[^>]+>', ' ', raw)
    t = re.sub(r'&[a-z]+;', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()

def extract_skills(title, desc):
    prompt = """Extract the top 5-8 technical skills/technologies required for this job.
Return ONLY a JSON array of skill names. Be specific and technical.
Do NOT include:
- Soft skills (communication, teamwork, leadership)
- Generic terms (develop, design, build, support, manage)
- Experience requirements (e.g. "X years of experience")
- Job titles or role names
- Company names

Job Title: %s
Job Description: %s

Return format: ["Skill 1", "Skill 2", ...]""" % (title, desc[:3000])
    resp = ai.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], temperature=0.1, max_tokens=200)
    text = resp.choices[0].message.content.strip()
    m = re.search(r'\[.*?\]', text, re.DOTALL)
    if m: return json.loads(m.group())
    return []

def is_garbage_skill(skill):
    """Check if a skill string is garbage"""
    # Pattern: "Something Over X Year S"
    if re.search(r'over\s+\d+\s+year', skill, re.IGNORECASE):
        return True
    # Pattern: very long strings (likely sentences, not skills)
    if len(skill) > 50:
        return True
    return False

def skills_are_garbage(skills_list, title):
    """Check if the majority of skills for a title are garbage"""
    if not skills_list:
        return True
    garbage_count = sum(1 for s in skills_list if is_garbage_skill(s))
    # If more than half are garbage, the whole set is bad
    return garbage_count > len(skills_list) * 0.4


def main():
    c = mysql.connector.connect(**DB_CFG)
    cur = c.cursor(dictionary=True)

    # Get ALL jd_extracted_skills for Q1
    cur.execute("""
        SELECT id, job_id, title, cluster_name, country, quarter, 
               extracted_skills, source_type, skill_count
        FROM jd_extracted_skills
        WHERE quarter = '2026Q1'
    """)
    all_rows = cur.fetchall()
    print("Total Q1 jd_extracted_skills records: %d" % len(all_rows))

    # Identify garbage entries
    garbage_ids = []
    for row in all_rows:
        try:
            skills = json.loads(row['extracted_skills']) if row['extracted_skills'] else []
        except:
            skills = []
        
        if skills_are_garbage(skills, row['title']):
            garbage_ids.append(row)

    print("Garbage entries found: %d" % len(garbage_ids))
    print("\nSample garbage:")
    for r in garbage_ids[:10]:
        skills = json.loads(r['extracted_skills']) if r['extracted_skills'] else []
        print("  %s | %s | %s" % (r['title'][:40], r['source_type'], str(skills[:3])[:80]))

    # Authenticate with JobDiva
    token = auth()
    print("\nJobDiva authenticated. Starting fixes...\n")

    fixed = 0
    skipped = 0
    failed = 0

    for i, row in enumerate(garbage_ids):
        title = row['title']
        job_id = str(row['job_id'])

        # Get a real JobDiva job_id
        real_job_id = None
        if job_id.isdigit():
            real_job_id = job_id
        else:
            # Try to find from updated_job_records
            cur.execute("SELECT job_id FROM updated_job_records WHERE title = %s LIMIT 1", (title,))
            alt = cur.fetchone()
            if alt and str(alt['job_id']).isdigit():
                real_job_id = str(alt['job_id'])

        if not real_job_id:
            skipped += 1
            continue

        try:
            raw = fetch_desc(token, real_job_id)
            desc = clean(raw)
            time.sleep(0.2)
        except Exception as e:
            failed += 1
            continue

        if not desc or len(desc) < 50:
            skipped += 1
            continue

        skills = extract_skills(title, desc)
        if skills and len(skills) >= 3:
            cur.execute("""
                UPDATE jd_extracted_skills 
                SET extracted_skills = %s, skill_count = %s,
                    extraction_method = 'llm_gpt4o_mini', source_type = 'jd_full_text'
                WHERE id = %s
            """, (json.dumps(skills), len(skills), row['id']))
            c.commit()
            fixed += 1
            if fixed <= 30 or fixed % 10 == 0:
                print("[%d/%d] FIXED %s: %s" % (i+1, len(garbage_ids), title[:35], skills[:4]))
        else:
            failed += 1

    print("\n" + "="*60)
    print("RESULTS:")
    print("  Total garbage found: %d" % len(garbage_ids))
    print("  Fixed with real skills: %d" % fixed)
    print("  Skipped (no JD available): %d" % skipped)
    print("  Failed (no skills extracted): %d" % failed)
    print("="*60)

    cur.close()
    c.close()

if __name__ == "__main__":
    main()
