import mysql.connector, json, os, re, sys
from dotenv import load_dotenv
from openai import AzureOpenAI

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))
DB_CFG = dict(host='localhost', port=3305, user='root', password='rootpassword', database='resume_processing')

ai = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
)

def extract_from_title(title):
    prompt = """Based on the job title below, list the top 5 technical skills/technologies typically required.
Return ONLY a JSON array of skill names. Be specific and technical.

Job Title: %s

Return format: ["Skill 1", "Skill 2", ...]""" % title
    resp = ai.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], temperature=0.2, max_tokens=200)
    text = resp.choices[0].message.content.strip()
    m = re.search(r'\[.*?\]', text, re.DOTALL)
    if m: return json.loads(m.group())
    return []

c = mysql.connector.connect(**DB_CFG)
cur = c.cursor(dictionary=True)

# Find remaining garbage
cur.execute("SELECT id, title, extracted_skills FROM jd_extracted_skills WHERE quarter = '2026Q1'")
garbage = []
for row in cur.fetchall():
    try:
        skills = json.loads(row['extracted_skills'])
    except:
        continue
    if any(re.search(r'Over \d+ Year', s) for s in skills):
        garbage.append(row)

print("Remaining garbage: %d" % len(garbage))

for row in garbage:
    title = row['title']
    skills = extract_from_title(title)
    if skills and len(skills) >= 3:
        cur.execute("UPDATE jd_extracted_skills SET extracted_skills = %s, skill_count = %s, extraction_method = 'llm_title_infer' WHERE id = %s",
                    (json.dumps(skills), len(skills), row['id']))
        c.commit()
        print("  FIXED %s: %s" % (title, skills))
    else:
        print("  SKIP %s" % title)

cur.close(); c.close()
print("Done")
