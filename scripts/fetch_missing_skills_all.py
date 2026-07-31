"""
Fetch JD descriptions from JobDiva for clusters missing skills,
then extract skills and insert into jd_extracted_skills.
Focus on clusters with higher demand first.
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

# Country code to full country mapping for updated_job_records
COUNTRY_MAP = {
    'US': 'US', 'IN': 'IN', 'AR': 'AR', 'BR': 'BR', 'CO': 'CO',
    'MX': 'MX', 'CA': 'CA', 'FR': 'FR', 'DE': 'DE', 'IT': 'IT',
    'PT': 'PT', 'AE': 'AE', 'AU': 'AU',
}

def main():
    conn = mysql.connector.connect(**DB_CFG)
    cur = conn.cursor(dictionary=True)

    # Find clusters without jd_extracted_skills
    cur.execute("""
        SELECT df.cluster_name, SUM(df.demand_predicted) as total_demand
        FROM demand_forecasts df
        LEFT JOIN (
            SELECT DISTINCT cluster_name COLLATE utf8mb4_0900_ai_ci as cluster_name FROM jd_extracted_skills
        ) es ON df.cluster_name = es.cluster_name
        WHERE es.cluster_name IS NULL
        GROUP BY df.cluster_name
        ORDER BY total_demand DESC
    """)
    missing = cur.fetchall()
    print(f"Clusters without extracted skills: {len(missing)}")
    
    # Only process clusters with demand >= 5
    targets = [r for r in missing if int(r['total_demand']) >= 5]
    print(f"Targeting {len(targets)} clusters with demand >= 5\n")

    token = authenticate()
    print("JobDiva authenticated\n")
    
    total_inserted = 0

    for cluster_row in targets:
        cluster_name = cluster_row['cluster_name']
        parts = cluster_name.split(' | ', 1)
        country_code = parts[0] if len(parts) == 2 else 'US'
        plain_role = parts[1] if len(parts) == 2 else cluster_name
        
        print(f"=== {cluster_name} (demand={int(cluster_row['total_demand'])}) ===")
        
        # Find job IDs from updated_job_records
        # For "LATAM & Others" clusters, search without country filter
        if 'LATAM' in country_code or len(country_code) > 3:
            cur.execute("""
                SELECT job_id, title, issue_date, country
                FROM updated_job_records
                WHERE role_cluster = %s
                ORDER BY issue_date DESC
                LIMIT 15
            """, (plain_role,))
        else:
            cur.execute("""
                SELECT job_id, title, issue_date, country
                FROM updated_job_records
                WHERE role_cluster = %s AND country = %s
                ORDER BY issue_date DESC
                LIMIT 15
            """, (plain_role, country_code))
        jobs = cur.fetchall()
        
        if not jobs:
            cur.execute("""
                SELECT job_id, title, issue_date, country
                FROM updated_job_records
                WHERE role_cluster = %s
                ORDER BY issue_date DESC
                LIMIT 15
            """, (plain_role,))
            jobs = cur.fetchall()
        
        if not jobs:
            print(f"  No job records found, skipping")
            continue
        
        print(f"  Found {len(jobs)} job records")
        
        fetched_count = 0
        for job in jobs:
            if fetched_count >= 5:
                break
                
            job_id = str(job['job_id'])
            title = job['title']
            
            # Check cache
            cur.execute("SELECT description FROM job_descriptions WHERE job_id = %s", (job_id,))
            cached = cur.fetchone()
            
            if cached and cached.get('description') and len(cached['description']) > 50:
                desc = cached['description']
            else:
                try:
                    raw = fetch_description(token, job_id)
                    desc = clean_description(raw)
                    if desc and len(desc) > 50:
                        if cached:
                            cur.execute("UPDATE job_descriptions SET description = %s WHERE job_id = %s", (desc, job_id))
                        else:
                            cur.execute("""
                                INSERT INTO job_descriptions (job_id, title, description)
                                VALUES (%s, %s, %s) ON DUPLICATE KEY UPDATE description = VALUES(description)
                            """, (job_id, title, desc))
                        conn.commit()
                        fetched_count += 1
                    else:
                        continue
                except Exception as e:
                    print(f"    Fetch error for {job_id}: {e}")
                    continue
                time.sleep(0.3)
            
            # Extract skills
            skills = extract_skills_llm(title, desc)
            if skills:
                # Determine quarter from issue_date
                issue = job['issue_date']
                if issue:
                    mon = issue.month
                    q = f"{issue.year}Q{(mon-1)//3+1}"
                else:
                    q = "2026Q1"
                
                job_country = job.get('country', country_code)
                if len(job_country) > 5:
                    job_country = job_country[:2]
                
                cur.execute("""
                    INSERT INTO jd_extracted_skills 
                    (job_id, cluster_name, company_name, country, title, extracted_skills, 
                     skill_count, extraction_method, source_type, quarter, extracted_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE extracted_skills = VALUES(extracted_skills)
                """, (
                    job_id, cluster_name, '', job_country, title,
                    json.dumps(skills), len(skills), 'llm_gpt4o_mini', 'full_text',
                    q, datetime.now()
                ))
                conn.commit()
                total_inserted += 1
                print(f"    {title}: {skills[:4]}")

    cur.close()
    conn.close()
    print(f"\n=== Done: {total_inserted} skill records inserted across {len(targets)} clusters ===")

if __name__ == "__main__":
    main()
