"""
Temporal Skill Evolution System
=================================
Phase 1: Build jd_skill_timeline table
Phase 2: Compute trends (rising/declining/emerging/stable)
Phase 3: Store aggregated skill trends per cluster/client/quarter

Uses issue_date from JDs to group skills into quarters.
"""
import sys
import re
import json
import mysql.connector
from collections import defaultdict
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)

DB_CONFIG = {
    "host": "localhost",
    "port": 3305,
    "database": "resume_processing",
    "user": "resume_user",
    "password": "resume_password",
}

SKILLS_TAXONOMY = {
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "golang",
    "rust", "ruby", "php", "swift", "kotlin", "scala", "matlab", "perl",
    "shell", "bash", "powershell", "sql", "plsql", "t-sql", "html", "css",
    "react", "angular", "vue", "node.js", "nodejs", "express", "django", "flask",
    "fastapi", "spring", "spring boot", "springboot", ".net", "asp.net",
    "laravel", "next.js", "nextjs", "svelte",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy",
    "hibernate", "jquery", "bootstrap", "tailwind",
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "k8s",
    "terraform", "ansible", "jenkins", "ci/cd", "cicd", "github actions",
    "cloudformation", "helm", "prometheus", "grafana", "datadog", "splunk",
    "mysql", "postgresql", "postgres", "mongodb", "redis", "elasticsearch",
    "cassandra", "dynamodb", "oracle", "sql server", "snowflake", "redshift",
    "bigquery", "hive", "spark", "kafka", "rabbitmq",
    "machine learning", "deep learning", "nlp", "natural language processing",
    "computer vision", "data science", "data engineering",
    "etl", "data pipeline", "airflow", "dbt", "tableau", "power bi",
    "hadoop", "flink", "databricks", "mlops", "generative ai", "llm",
    "git", "github", "gitlab", "jira", "agile", "scrum", "devops", "sre",
    "linux", "unix", "rest api", "restful", "graphql", "grpc", "websocket",
    "microservices", "serverless", "lambda",
    "cybersecurity", "penetration testing", "encryption", "oauth", "saml",
    "devsecops", "compliance",
    "embedded systems", "embedded linux", "rtos", "firmware", "fpga",
    "verilog", "vhdl", "arm", "misra c", "can bus", "iot",
    "secure boot", "device drivers", "yocto", "buildroot",
    "ios", "android", "react native", "flutter",
    "selenium", "cypress", "jest", "junit", "pytest", "playwright",
    "tdd", "design patterns", "clean architecture",
}

SKIP_TERMS = {
    "develop", "design", "web", "website", "work", "team", "support",
    "manage", "build", "create", "test", "project", "system", "data",
    "application", "software", "service", "experience", "years",
    "web or website", "null", "linkedin", "skills to be assigned",
    "senior", "junior", "lead", "manager", "engineer", "developer",
    "consultant", "analyst", "associate", "specialist", "coordinator",
}


def get_quarter(date_val):
    """Convert a date to quarter string like '2025-Q3'."""
    if not date_val:
        return None
    if isinstance(date_val, str):
        try:
            date_val = datetime.strptime(date_val[:10], "%Y-%m-%d")
        except:
            return None
    q = (date_val.month - 1) // 3 + 1
    return f"{date_val.year}-Q{q}"


def parse_boolean_skills(raw):
    """Parse '(JAVA over 3 year(s)) AND (AWS)' format."""
    if not raw or len(raw) < 5:
        return []
    terms = re.findall(r'\(([^)]+)\)', raw)
    skills = []
    for term in terms:
        clean = re.sub(r'\s+over\s+\d+\s+year\(s\)', '', term, flags=re.IGNORECASE).strip()
        if clean and len(clean) >= 2 and not re.match(r'^\d+$', clean):
            cl = clean.lower()
            if cl in SKIP_TERMS:
                continue
            if re.match(r'^[A-Z][a-z]+ [A-Z][a-z]+$', clean):  # Skip names
                continue
            if len(clean) > 100:  # Too long
                continue
            if cl in SKILLS_TAXONOMY or len(clean) > 2:
                skills.append(clean.title() if len(clean) > 3 else clean.upper())
    return skills[:8]


def main():
    print("=" * 70, flush=True)
    print("TEMPORAL SKILL EVOLUTION — Building Timeline", flush=True)
    print("=" * 70, flush=True)

    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor(dictionary=True)

    # ═══ Step 1: Create tables ═══════════════════════════════════════════════
    print("\n[1/5] Creating tables...", flush=True)

    cur.execute("DROP TABLE IF EXISTS jd_skill_timeline")
    cur.execute("""
        CREATE TABLE jd_skill_timeline (
            id INT AUTO_INCREMENT PRIMARY KEY,
            cluster_name VARCHAR(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
            company_name VARCHAR(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
            country VARCHAR(10) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
            quarter VARCHAR(10) NOT NULL,
            skill VARCHAR(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
            frequency INT DEFAULT 1,
            jd_count INT DEFAULT 1,
            INDEX idx_cluster_quarter (cluster_name, quarter),
            INDEX idx_skill (skill),
            INDEX idx_company (company_name, quarter),
            UNIQUE KEY uq_entry (cluster_name, company_name, country, quarter, skill)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)

    cur.execute("DROP TABLE IF EXISTS skill_trends")
    cur.execute("""
        CREATE TABLE skill_trends (
            id INT AUTO_INCREMENT PRIMARY KEY,
            cluster_name VARCHAR(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
            company_name VARCHAR(200) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci DEFAULT '_ALL_',
            country VARCHAR(10) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
            skill VARCHAR(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci NOT NULL,
            trend_category VARCHAR(20) NOT NULL,
            trending_score FLOAT DEFAULT 0,
            current_frequency INT DEFAULT 0,
            previous_frequency INT DEFAULT 0,
            first_seen VARCHAR(10),
            last_seen VARCHAR(10),
            quarters_active INT DEFAULT 0,
            computed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            INDEX idx_cluster (cluster_name),
            INDEX idx_trend (trend_category),
            INDEX idx_company_cluster (company_name, cluster_name),
            UNIQUE KEY uq_trend (cluster_name, company_name, country, skill)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """)
    conn.commit()
    print("  ✓ Tables created: jd_skill_timeline, skill_trends", flush=True)

    # ═══ Step 2: Build timeline from updated_job_records ═════════════════════
    print("\n[2/5] Building skill timeline from job records...", flush=True)

    # Get all JDs with issue_date, skills, and cluster assignment
    cur.execute("""
        SELECT title, company_name, country, skills, role_cluster, issue_date
        FROM updated_job_records
        WHERE role_cluster IS NOT NULL
        AND issue_date IS NOT NULL
        AND issue_date >= '2020-01-01'
        AND skills IS NOT NULL AND skills != '' AND skills != 'Null'
        AND LENGTH(skills) > 10
        AND skills REGEXP '\\\\('
    """)
    records = cur.fetchall()
    print(f"  Found {len(records)} job records with dates + skills + clusters", flush=True)

    # Build timeline: cluster × company × country × quarter × skill → frequency
    timeline = defaultdict(lambda: defaultdict(int))  # key → skill → count
    jd_counts = defaultdict(int)  # key → total JDs in that quarter

    for rec in records:
        quarter = get_quarter(rec["issue_date"])
        if not quarter:
            continue

        cluster = f"{rec['country'] or 'US'} | {rec['role_cluster']}"
        company = rec["company_name"] or "Unknown"
        country = rec["country"] or "US"

        skills = parse_boolean_skills(rec["skills"])
        if not skills:
            continue

        # Track per-company
        key_company = (cluster, company, country, quarter)
        # Track all-companies aggregate
        key_all = (cluster, "_ALL_", country, quarter)

        for skill in skills:
            timeline[key_company][skill] += 1
            timeline[key_all][skill] += 1

        jd_counts[key_company] += 1
        jd_counts[key_all] += 1

    print(f"  Processed into {len(timeline)} cluster×company×quarter combinations", flush=True)

    # ═══ Step 3: Also add from full-text extracted skills ════════════════════
    print("\n[3/5] Adding full-text extracted skills to timeline...", flush=True)

    cur.execute("""
        SELECT jes.cluster_name, jes.company_name, jes.country, jes.extracted_skills,
               jd.issue_date
        FROM jd_extracted_skills jes
        JOIN job_descriptions jd ON jes.job_id = jd.job_id
        WHERE jes.source_type = 'jd_full_text'
        AND jd.issue_date IS NOT NULL
        AND jd.issue_date >= '2020-01-01'
    """)
    fulltext_records = cur.fetchall()
    print(f"  {len(fulltext_records)} full-text extracted records with dates", flush=True)

    for rec in fulltext_records:
        quarter = get_quarter(rec["issue_date"])
        if not quarter or not rec["cluster_name"]:
            continue

        cluster = rec["cluster_name"]
        company = rec["company_name"] or "Unknown"
        country = rec["country"] or "US"

        try:
            skills = json.loads(rec["extracted_skills"])
        except:
            continue

        key_company = (cluster, company, country, quarter)
        key_all = (cluster, "_ALL_", country, quarter)

        for skill in skills:
            if skill.lower() not in SKIP_TERMS and len(skill) <= 100:
                # Full-text gets 3x weight
                timeline[key_company][skill] += 3
                timeline[key_all][skill] += 3

        jd_counts[key_company] += 1
        jd_counts[key_all] += 1

    # ═══ Step 4: Insert into jd_skill_timeline ════════════════════════════════
    print("\n[4/5] Inserting into jd_skill_timeline...", flush=True)

    insert_count = 0
    batch = []
    for (cluster, company, country, quarter), skills in timeline.items():
        jd_count = jd_counts[(cluster, company, country, quarter)]
        for skill, freq in skills.items():
            batch.append((cluster, company, country, quarter, skill, freq, jd_count))
            if len(batch) >= 1000:
                cur.executemany("""
                    INSERT INTO jd_skill_timeline
                    (cluster_name, company_name, country, quarter, skill, frequency, jd_count)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE frequency = VALUES(frequency), jd_count = VALUES(jd_count)
                """, batch)
                insert_count += len(batch)
                batch = []

    if batch:
        cur.executemany("""
            INSERT INTO jd_skill_timeline
            (cluster_name, company_name, country, quarter, skill, frequency, jd_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE frequency = VALUES(frequency), jd_count = VALUES(jd_count)
        """, batch)
        insert_count += len(batch)

    conn.commit()
    print(f"  ✓ Inserted {insert_count} timeline records", flush=True)

    # ═══ Step 5: Compute trends ══════════════════════════════════════════════
    print("\n[5/5] Computing skill trends...", flush=True)

    # Get all unique cluster/company/country combos
    cur.execute("""
        SELECT DISTINCT cluster_name, company_name, country
        FROM jd_skill_timeline
    """)
    combos = cur.fetchall()
    print(f"  {len(combos)} cluster/company/country combos to analyze", flush=True)

    # Define "recent" vs "previous" quarters
    # Recent: last 4 quarters (2025-Q3 to 2026-Q2)
    # Previous: 4 quarters before that (2024-Q3 to 2025-Q2)
    recent_quarters = {"2025-Q3", "2025-Q4", "2026-Q1", "2026-Q2"}
    previous_quarters = {"2024-Q3", "2024-Q4", "2025-Q1", "2025-Q2"}
    emerging_cutoff = {"2025-Q4", "2026-Q1", "2026-Q2"}  # first seen in last 3 quarters

    trend_batch = []
    for combo in combos:
        cluster = combo["cluster_name"]
        company = combo["company_name"]
        country = combo["country"]

        # Get all skills for this combo across quarters
        cur.execute("""
            SELECT skill, quarter, frequency
            FROM jd_skill_timeline
            WHERE cluster_name = %s AND company_name = %s AND country = %s
            ORDER BY quarter
        """, (cluster, company, country))
        rows = cur.fetchall()

        # Build skill → {quarter: freq} map
        skill_history = defaultdict(dict)
        for row in rows:
            skill_history[row["skill"]][row["quarter"]] = row["frequency"]

        # Compute trend for each skill
        for skill, quarters_data in skill_history.items():
            all_quarters = sorted(quarters_data.keys())
            first_seen = all_quarters[0]
            last_seen = all_quarters[-1]
            quarters_active = len(all_quarters)

            # Sum frequencies in recent vs previous periods
            recent_freq = sum(quarters_data.get(q, 0) for q in recent_quarters)
            prev_freq = sum(quarters_data.get(q, 0) for q in previous_quarters)

            # Trending score
            if prev_freq > 0:
                trending_score = (recent_freq / prev_freq) - 1.0
            elif recent_freq > 0:
                trending_score = 2.0  # New skill, infinite growth
            else:
                trending_score = 0.0

            # Classify
            if first_seen in emerging_cutoff and quarters_active <= 3:
                category = "emerging"
            elif trending_score > 0.3:
                category = "rising"
            elif trending_score < -0.3:
                category = "declining"
            else:
                category = "stable"

            trend_batch.append((
                cluster, company, country, skill, category,
                round(trending_score, 2), recent_freq, prev_freq,
                first_seen, last_seen, quarters_active
            ))

    # Insert trends
    if trend_batch:
        cur.executemany("""
            INSERT INTO skill_trends
            (cluster_name, company_name, country, skill, trend_category,
             trending_score, current_frequency, previous_frequency,
             first_seen, last_seen, quarters_active)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                trend_category = VALUES(trend_category),
                trending_score = VALUES(trending_score),
                current_frequency = VALUES(current_frequency),
                previous_frequency = VALUES(previous_frequency),
                computed_at = NOW()
        """, trend_batch)
        conn.commit()

    print(f"  ✓ Computed {len(trend_batch)} skill trends", flush=True)

    # ═══ Results ═════════════════════════════════════════════════════════════
    print(f"\n{'='*70}", flush=True)
    print("RESULTS", flush=True)
    print(f"{'='*70}", flush=True)

    cur.execute("SELECT COUNT(*) as cnt FROM jd_skill_timeline")
    print(f"\nTimeline records: {cur.fetchone()['cnt']}", flush=True)

    cur.execute("SELECT COUNT(*) as cnt FROM skill_trends")
    print(f"Trend records: {cur.fetchone()['cnt']}", flush=True)

    cur.execute("""
        SELECT trend_category, COUNT(*) as cnt
        FROM skill_trends WHERE company_name = '_ALL_'
        GROUP BY trend_category ORDER BY cnt DESC
    """)
    print(f"\nTrend categories (all-companies aggregate):", flush=True)
    for r in cur.fetchall():
        print(f"  {r['trend_category']}: {r['cnt']} skills", flush=True)

    # Show examples
    for category in ['emerging', 'rising', 'declining', 'stable']:
        cur.execute("""
            SELECT cluster_name, skill, trending_score, current_frequency, first_seen
            FROM skill_trends
            WHERE trend_category = %s AND company_name = '_ALL_'
            ORDER BY ABS(trending_score) DESC
            LIMIT 5
        """, (category,))
        rows = cur.fetchall()
        if rows:
            print(f"\n  Top {category.upper()} skills:", flush=True)
            for r in rows:
                score = f"+{r['trending_score']:.0%}" if r['trending_score'] > 0 else f"{r['trending_score']:.0%}"
                print(f"    {r['cluster_name']}: {r['skill']} ({score}, freq={r['current_frequency']}, since {r['first_seen']})", flush=True)

    cur.close()
    conn.close()
    print(f"\n✓ Temporal Skill Evolution system built!", flush=True)


if __name__ == "__main__":
    main()
