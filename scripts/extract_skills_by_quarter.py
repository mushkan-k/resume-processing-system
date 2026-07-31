"""
Extract skills per cluster per quarter.
- Groups JDs by (region, cluster, quarter)
- Extracts skills from each JD individually
- Stores with quarter info so API can serve quarter-specific skills
"""
import os, sys, re, json, time
import pandas as pd
import mysql.connector
from collections import Counter
from dotenv import load_dotenv

load_dotenv(r"c:\Users\serveradmin\Desktop\resume-processing-system\.env")
DB = dict(host='localhost', port=3305, user='root', password='rootpassword', database='resume_processing')

SKIP_WORDS = {
    'engineer','developer','manager','analyst','technician','specialist',
    'architect','administrator','coordinator','designer','consultant',
    'lead','senior','junior','principal','director','associate','intern',
    'contractor','supervisor','officer','president','scientist','researcher',
    'builder','auditor','inspector','planner','programmer','tester',
    'experience','knowledge','ability','understanding','skills','team',
    'work','working','role','position','job','candidate','company','client',
    'project','projects','support','process','processes','system','systems',
    'data','information','provide','ensure','requirements','required',
    'including','related','based','using','used','years','year',
    'strong','good','excellent','preferred','must','able',
    'skills to be assigned','tested','material','materials','equipment',
    'lab','laboratory','documentation','protocols','investigation',
    'product','manufacture','packaging','drug',
}

def is_real_skill(term):
    t = term.strip().lower()
    if len(t) < 2 or len(t) > 50: return False
    t = re.sub(r'\s*\{[^}]*\}\s*', '', t).strip()
    if len(t) < 2: return False
    if t in SKIP_WORDS: return False
    words = set(t.split())
    if words and words.issubset(SKIP_WORDS): return False
    role_suffixes = {'engineer','developer','manager','analyst','technician','specialist',
                     'architect','consultant','designer','tester','auditor','coordinator',
                     'administrator','assistant','writer','planner','officer'}
    w_list = t.split()
    if len(w_list) >= 2 and w_list[-1] in role_suffixes: return False
    return True

def normalise_skill(term):
    t = re.sub(r'\s*\{[^}]*\}\s*', '', term).strip()
    tl = t.lower()
    acronyms = {'aws','gcp','sql','api','ci/cd','iam','plc','sap','crm','erp',
                'hplc','eln','gmp','cgmp','gdp','fda','sop','capa','qa','qc',
                'html','css','vba','rpa','iot','fpga','soc','siem','etl','sdk',
                'json','xml','yaml','rest','pmp','itil','sox','hipaa','gdpr','iso'}
    if tl in acronyms: return tl.upper()
    lower_style = {'python','java','javascript','typescript','kotlin','scala',
                   'ruby','perl','php','bash','golang','rust','swift','dart','lua',
                   'docker','kubernetes','terraform','ansible','jenkins',
                   'react','angular','vue','node.js','express','django','flask',
                   'spring','hadoop','spark','kafka','redis','nginx','apache',
                   'selenium','junit','pytest','cypress','jest','jmeter',
                   'git','github','gitlab','bitbucket',
                   'agile','scrum','kanban','jira','confluence','lean',
                   'salesforce','servicenow','workday','sharepoint','tableau',
                   'snowflake','databricks','airflow','looker',
                   'mysql','postgresql','mongodb','oracle','cassandra','elasticsearch',
                   'vmware','openstack','linux','unix','windows',
                   'creo','solidworks','autocad','catia','ansys','simulink','matlab',
                   'smartsheet','excel','powerpoint','figma'}
    if tl in lower_style: return tl.title()
    return t.title()

def parse_skills_clean(text):
    if not text or not str(text).strip() or str(text).strip() == 'nan': return []
    clean = str(text).replace('(', ' ').replace(')', ' ')
    terms = re.split(r'\s+(?:AND|OR)\s+', clean, flags=re.IGNORECASE)
    skills = []
    seen = set()
    for t in terms:
        t = re.sub(r'^(?:And|Or)', '', t.strip()).strip()
        if not is_real_skill(t): continue
        n = normalise_skill(t)
        nl = n.lower()
        if nl not in seen and len(n) >= 2:
            seen.add(nl)
            skills.append(n)
    return skills

def extract_skills_llm(client, model, description, title):
    prompt = f"""You are a technical recruiter analyzing a job description. Extract ALL required and preferred skills, technologies, and qualifications.

Return a JSON array of skill strings. Include:
1. Programming languages & frameworks (Python, Java, React, Spring Boot, .NET, etc.)
2. Cloud & infrastructure (AWS, Azure, GCP, Docker, Kubernetes, Terraform, etc.)
3. Tools & software platforms (SAP, Salesforce, DeltaV, LIMS, Jira, ServiceNow, etc.)
4. Data & analytics (SQL, Tableau, Power BI, Spark, Machine Learning, etc.)
5. Domain/industry skills (chromatography, cell culture, GMP, financial modeling, upstream/downstream processing, etc.)
6. Methodologies & processes (Agile, Scrum, Six Sigma, SDLC, CI/CD, DevOps, etc.)
7. Certifications & standards (PMP, AWS Certified, ISO 9001, 21 CFR Part 11, GAMP, etc.)
8. Hardware & equipment (PLC, bioreactors, SCADA, DCS, embedded systems, etc.)

Rules:
- Extract up to 25 skills
- Be SPECIFIC — prefer "DeltaV DCS" over "control systems", "AWS Lambda" over "cloud"
- Include both required AND preferred/nice-to-have skills
- NORMALISE variations to ONE canonical form:
  * "JS" / "Javascript" / "JavaScript" → "JavaScript"
  * "K8s" / "Kubernetes" / "k8" → "Kubernetes"  
  * "React.js" / "ReactJS" / "React" → "React"
  * "CI/CD" / "CICD" / "CI-CD" → "CI/CD"
  * "Postgres" / "PostgreSQL" / "psql" → "PostgreSQL"
  * "C Sharp" / "C#" / "CSharp" → "C#"
  * "DotNet" / ".NET" / "Dot Net" → ".NET"
  * "ML" / "Machine Learning" → "Machine Learning"
  * "DL" / "Deep Learning" → "Deep Learning"
  * "NLP" / "Natural Language Processing" → "NLP"
  * "Tableau" not "tableau" — always use proper casing
- Do NOT include soft skills (communication, teamwork, leadership, problem-solving)
- Do NOT include job titles or role names (Engineer, Developer, Manager)
- Do NOT include generic words (experience, knowledge, years, understanding, ability)
- Do NOT include company names or locations

Job Title: {title}

Job Description:
{description[:5000]}"""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0, max_tokens=500,
                timeout=30,
            )
            text = resp.choices[0].message.content.strip()
            if text.startswith('['): return json.loads(text)
            m = re.search(r'\[.*\]', text, re.DOTALL)
            return json.loads(m.group()) if m else []
        except Exception as e:
            if attempt < 2:
                time.sleep(2 * (attempt + 1))
                continue
            print(f"    LLM error (3 retries): {e}")
            return []

def main():
    print("=" * 65)
    print("Skill Extraction — Per Cluster Per Quarter")
    print("=" * 65)
    
    conn = mysql.connector.connect(**DB)
    cur = conn.cursor(dictionary=True)
    
    # Add quarter column if not exists
    try:
        cur.execute("ALTER TABLE jd_extracted_skills ADD COLUMN quarter VARCHAR(10) DEFAULT NULL AFTER source_type")
        conn.commit()
        print("  Added 'quarter' column to jd_extracted_skills")
    except:
        conn.rollback()
        print("  'quarter' column already exists")
    
    # Make job_id non-unique (we'll have multiple quarters)
    try:
        cur.execute("ALTER TABLE jd_extracted_skills DROP INDEX job_id")
        conn.commit()
        print("  Dropped unique index on job_id")
    except:
        conn.rollback()
    
    # Load data
    print("\nLoading data...")
    df = pd.read_pickle(r"c:\Users\serveradmin\Desktop\resume-processing-system\data\clean_42k_v1.pkl")
    df['issue_date'] = pd.to_datetime(df['issue_date'])
    
    t2c = pd.read_pickle(r"c:\Users\serveradmin\Desktop\resume-processing-system\data\title_to_cluster.pkl")
    title_map = dict(zip(t2c['raw_title'].str.lower().str.strip(), t2c['role_cluster']))
    
    # ALL 2026 data (Q1 through Q3 if available)
    mask = (df['issue_date'] >= '2026-01-01') & (df['issue_date'] < '2026-10-01')
    all_jds = df[mask].copy()
    all_jds['cluster'] = all_jds['title'].str.lower().str.strip().map(title_map)
    all_jds = all_jds.dropna(subset=['cluster'])
    all_jds['quarter'] = all_jds['issue_date'].dt.to_period('Q').astype(str)
    
    print(f"  Total JDs: {len(all_jds)}")
    print(f"  Quarters: {sorted(all_jds['quarter'].unique())}")
    print(f"  Regions: {sorted(all_jds['region'].unique())}")
    
    # Get descriptions from DB
    cur.execute("""SELECT job_id, title, country, description FROM job_descriptions 
                   WHERE description IS NOT NULL AND LENGTH(description) > 100""")
    db_descs = cur.fetchall()
    desc_by_key = {}
    for r in db_descs:
        key = (r['country'], r['title'].lower().strip())
        if key not in desc_by_key or len(str(r['description'])) > len(str(desc_by_key[key].get('description',''))):
            desc_by_key[key] = r
    print(f"  JDs with descriptions: {len(desc_by_key)}")
    
    # Azure OpenAI
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    )
    model = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
    
    # Group by region + cluster + quarter
    groups = all_jds.groupby(['region', 'cluster', 'quarter'])
    print(f"\n  {len(groups)} (region x cluster x quarter) combos\n")
    
    # Clear old data
    if '--dry-run' not in sys.argv:
        cur.execute("DELETE FROM jd_extracted_skills")
        conn.commit()
        print("  Cleared old data\n")
    
    all_results = []
    llm_total = 0
    parse_total = 0
    
    for gi, ((region, cluster, quarter), gdf) in enumerate(groups):
        cluster_name = f"{region} | {cluster}"
        skills_this_group = Counter()
        results = []
        llm_this = 0
        parse_this = 0
        
        for _, row in gdf.iterrows():
            title_lower = row['title'].lower().strip()
            company = str(row.get('company_name', ''))
            skills = []
            method = 'nlp_taxonomy_v2'
            source = 'jd_skills_field'
            job_id = f"pkl_{abs(hash(title_lower + company + region + quarter)) % 1000000:06d}"
            
            # Try LLM
            desc_key = (region, title_lower)
            if desc_key in desc_by_key:
                desc_text = str(desc_by_key[desc_key]['description'])
                if len(desc_text) > 100:
                    skills = extract_skills_llm(client, model, desc_text, row['title'])
                    if skills:
                        method = 'llm_gpt4o_mini'
                        source = 'jd_full_text'
                        job_id = str(desc_by_key[desc_key]['job_id'])
                        llm_this += 1
                        time.sleep(0.12)
            
            # Fallback: parse skills_clean
            if not skills:
                skills = parse_skills_clean(str(row.get('skills_clean', '')))
                if skills:
                    parse_this += 1
            
            if not skills:
                continue
            
            for s in skills:
                skills_this_group[s] += 1
            
            results.append((
                job_id, cluster_name, company, region, row['title'],
                json.dumps(skills), len(skills), method, source, quarter
            ))
        
        if results:
            top3 = ", ".join(f"{s}({c})" for s, c in skills_this_group.most_common(3))
            if llm_this > 0 or len(results) >= 5:
                print(f"  [{gi+1:3d}] {cluster_name:<35} {quarter} | {len(gdf):3d} JDs | {len(results):3d} skills | LLM:{llm_this} NLP:{parse_this} | {top3}")
        
        all_results.extend(results)
        llm_total += llm_this
        parse_total += parse_this
    
    print(f"\n{'─'*65}")
    print(f"Total: {len(all_results)} skill records")
    print(f"  LLM: {llm_total}  |  NLP: {parse_total}")
    
    if '--dry-run' in sys.argv:
        print("\n[DRY RUN]")
        return
    
    # Write to DB
    print("\nWriting to DB...")
    insert_sql = """INSERT INTO jd_extracted_skills 
        (job_id, cluster_name, company_name, country, title, extracted_skills, skill_count, extraction_method, source_type, quarter)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    
    seen_ids = set()
    batch = []
    for r in all_results:
        jid = r[0]
        if jid in seen_ids:
            jid = jid + f"_{len(seen_ids)}"
        seen_ids.add(jid)
        batch.append((jid,) + r[1:])
    
    for i in range(0, len(batch), 200):
        cur.executemany(insert_sql, batch[i:i+200])
        conn.commit()
    print(f"  ✓ Inserted {len(batch)} rows")
    
    # Verify
    cur.execute("""SELECT quarter, country, COUNT(*) cnt, COUNT(DISTINCT cluster_name) clusters
                   FROM jd_extracted_skills GROUP BY quarter, country ORDER BY quarter, country""")
    print("\n  By quarter x region:")
    for r in cur.fetchall():
        print(f"    {r['quarter']} {r['country']}: {r['cnt']} records, {r['clusters']} clusters")
    
    # Show sample: US | Software Engineer by quarter
    print("\n  Sample — US | Software Engineer skills by quarter:")
    for q in ['2026Q1', '2026Q2']:
        cur.execute("""SELECT extracted_skills FROM jd_extracted_skills 
                       WHERE cluster_name = 'US | Software Engineer' AND quarter = %s 
                       ORDER BY CASE source_type WHEN 'jd_full_text' THEN 1 ELSE 2 END, skill_count DESC
                       LIMIT 50""", (q,))
        rows = cur.fetchall()
        freq = Counter()
        for row in rows:
            for s in json.loads(row['extracted_skills']):
                freq[s] += 1
        top5 = [f"{s}({c})" for s, c in freq.most_common(5)]
        print(f"    {q}: {len(rows)} JDs → {', '.join(top5)}")
    
    conn.close()
    print("\nDone! ✓")

if __name__ == '__main__':
    main()
