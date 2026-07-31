import mysql.connector, json, os, re, requests, time, traceback
from dotenv import load_dotenv
from openai import AzureOpenAI
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))
print("KEY loaded:", bool(os.getenv("AZURE_OPENAI_KEY")))
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

def extract(title, desc):
    prompt = """Extract the top 5-8 technical skills/technologies required for this job.
Return ONLY a JSON array of skill names. Be specific and technical.
Do not include soft skills, generic terms, or company names.
Do NOT include experience requirements like "X years" or "Over X Year".

Job Title: %s
Job Description: %s

Return format: ["Skill 1", "Skill 2", ...]""" % (title, desc[:3000])
    resp = ai.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], temperature=0.1, max_tokens=200)
    text = resp.choices[0].message.content.strip()
    m = re.search(r'\[.*?\]', text, re.DOTALL)
    if m: return json.loads(m.group())
    return []

c = mysql.connector.connect(**DB_CFG)
cur = c.cursor(dictionary=True)

# Find ALL entries where ANY skill contains "Over \d+ Year" pattern
cur.execute("""
    SELECT id, job_id, title, cluster_name, country, quarter, extracted_skills, source_type
    FROM jd_extracted_skills
    WHERE quarter = '2026Q1'
""")
all_rows = cur.fetchall()

garbage = []
for row in all_rows:
    try:
        skills = json.loads(row['extracted_skills'])
    except:
        continue
    # Check if ANY skill has "Over X Year" pattern
    has_garbage = any(re.search(r'Over \d+ Year', s) for s in skills)
    if has_garbage:
        garbage.append(row)

print("Total Q1 records: %d" % len(all_rows))
print("Records with 'Over X Year' garbage: %d" % len(garbage))

token = auth()
print("Authenticated. Fixing...\n")

fixed = 0
skipped = 0
for i, row in enumerate(garbage):
    title = row['title']
    job_id = str(row['job_id'])
    
    # Get real job_id if synthetic
    if not job_id.isdigit():
        cur.execute("SELECT job_id FROM updated_job_records WHERE title = %s LIMIT 1", (title,))
        alt = cur.fetchone()
        if alt and str(alt['job_id']).isdigit():
            job_id = str(alt['job_id'])
        else:
            skipped += 1
            continue
    
    try:
        raw = fetch_desc(token, job_id)
        desc = clean(raw)
        time.sleep(0.3)
    except:
        skipped += 1
        continue
    
    if not desc or len(desc) < 50:
        skipped += 1
        continue
    
    skills = extract(title, desc)
    if skills and len(skills) >= 3:
        cur.execute("""
            UPDATE jd_extracted_skills 
            SET extracted_skills = %s, skill_count = %s, 
                extraction_method = 'llm_gpt4o_mini', source_type = 'jd_full_text'
            WHERE id = %s
        """, (json.dumps(skills), len(skills), row['id']))
        c.commit()
        fixed += 1
        if fixed <= 20 or fixed % 20 == 0:
            print("[%d/%d] FIXED %s: %s" % (fixed, len(garbage), title, skills[:4]))
    else:
        skipped += 1

print("\n" + "="*50)
print("Fixed: %d / %d" % (fixed, len(garbage)))
print("Skipped: %d" % skipped)
cur.close(); c.close()
