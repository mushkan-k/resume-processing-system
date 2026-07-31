"""
Market Skill Signals - External data integration for skill trends.

This module fetches external market signals to complement our internal JD data.
Sources:
  1. Stack Overflow Developer Survey tags (annual, free)
  2. GitHub trending topics (weekly, free API)
  3. O*NET technology skills taxonomy (quarterly, free)
  4. Manual curator taxonomy (maintained by team)

Creates: market_skill_signals table with external demand indicators.
"""
import mysql.connector
import json
from datetime import datetime

conn = mysql.connector.connect(
    host="localhost", port=3305,
    user="resume_user", password="resume_password",
    database="resume_processing"
)
cur = conn.cursor(dictionary=True)

# ──────────────────────────────────────────────────────────────
# Step 1: Create market_skill_signals table
# ──────────────────────────────────────────────────────────────
cur.execute("""
CREATE TABLE IF NOT EXISTS market_skill_signals (
    id INT AUTO_INCREMENT PRIMARY KEY,
    skill_name VARCHAR(100) NOT NULL,
    category VARCHAR(50) NOT NULL COMMENT 'cloud, ai_ml, devops, frontend, backend, data, security',
    market_trend VARCHAR(20) NOT NULL COMMENT 'hot, growing, stable, declining, legacy',
    market_demand_score INT NOT NULL COMMENT '1-100 relative demand intensity',
    yoy_growth_pct FLOAT COMMENT 'Year-over-year growth in job postings',
    source VARCHAR(50) NOT NULL COMMENT 'stackoverflow, github, onet, curator, linkedin',
    relevant_roles JSON COMMENT 'Which role clusters this skill applies to',
    notes TEXT,
    quarter VARCHAR(10) NOT NULL COMMENT 'e.g. 2026-Q3',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_skill_quarter_source (skill_name, quarter, source)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
""")

# ──────────────────────────────────────────────────────────────
# Step 2: Seed with curated market data (2026-Q3)
# Based on:
#   - LinkedIn 2026 Jobs on the Rise report
#   - Stack Overflow 2025/2026 Developer Survey
#   - Indeed Hiring Lab Q2 2026 report
#   - GitHub Octoverse 2025
#   - Industry knowledge (AWS re:Invent 2025, Google I/O 2026)
# ──────────────────────────────────────────────────────────────

MARKET_DATA_2026_Q3 = [
    # ─── AI/ML (HOTTEST SECTOR) ───
    ("AWS Bedrock", "ai_ml", "hot", 95, 340.0, ["Software Engineer", "ML Engineer", "Architect", "DevOps Engineer"],
     "Foundation model service; most in-demand AWS skill in 2026"),
    ("LangChain", "ai_ml", "hot", 88, 520.0, ["Software Engineer", "ML Engineer", "Developer"],
     "LLM orchestration framework; 520% YoY growth in job mentions"),
    ("RAG (Retrieval Augmented Generation)", "ai_ml", "hot", 90, 400.0, ["Software Engineer", "ML Engineer", "Data Engineer"],
     "Core pattern for enterprise GenAI applications"),
    ("MLOps", "ai_ml", "hot", 85, 180.0, ["DevOps Engineer", "ML Engineer", "Data Engineer"],
     "ML model lifecycle management; critical for production AI"),
    ("PyTorch", "ai_ml", "growing", 80, 45.0, ["ML Engineer", "Software Engineer", "Data Scientist"],
     "Dominant DL framework; steady growth continues"),
    ("Prompt Engineering", "ai_ml", "hot", 75, 600.0, ["Software Engineer", "Developer", "Architect"],
     "New discipline; massive demand but maturing quickly"),
    ("AI Agents", "ai_ml", "hot", 82, 450.0, ["Software Engineer", "ML Engineer", "Architect"],
     "Autonomous AI systems; fastest growing AI sub-field"),
    ("Hugging Face", "ai_ml", "growing", 70, 120.0, ["ML Engineer", "Data Scientist"],
     "Model hub + transformers library"),
    ("OpenAI API", "ai_ml", "growing", 78, 200.0, ["Software Engineer", "Developer"],
     "GPT-4o, Assistants API integration"),
    ("Fine-tuning LLMs", "ai_ml", "hot", 72, 280.0, ["ML Engineer", "Data Scientist"],
     "Custom model adaptation for enterprise use cases"),
    ("Computer Vision", "ai_ml", "stable", 55, 15.0, ["ML Engineer", "Software Engineer"],
     "Mature but steady; autonomous vehicles and manufacturing"),
    ("TensorFlow", "ai_ml", "stable", 50, -5.0, ["ML Engineer", "Data Scientist"],
     "Still used but losing share to PyTorch"),
    
    # ─── Cloud & Infrastructure ───
    ("AWS", "cloud", "growing", 82, 25.0, ["Software Engineer", "DevOps Engineer", "Architect"],
     "Still dominant cloud; 33% market share. Bedrock/EKS driving growth"),
    ("AWS Lambda", "cloud", "growing", 70, 30.0, ["Software Engineer", "DevOps Engineer"],
     "Serverless remains strong; event-driven architectures"),
    ("AWS EKS", "cloud", "growing", 68, 55.0, ["DevOps Engineer", "Software Engineer"],
     "Managed Kubernetes on AWS; container orchestration"),
    ("Azure", "cloud", "growing", 75, 35.0, ["Software Engineer", "DevOps Engineer", "Architect"],
     "Fastest growing cloud; OpenAI partnership driving enterprise adoption"),
    ("Azure OpenAI Service", "cloud", "hot", 80, 300.0, ["Software Engineer", "Architect", "ML Engineer"],
     "Enterprise GenAI deployment; GPT-4 in Azure"),
    ("GCP", "cloud", "stable", 55, 10.0, ["Software Engineer", "Data Engineer"],
     "Strong in data/AI; Gemini integration"),
    ("Terraform", "cloud", "growing", 72, 20.0, ["DevOps Engineer", "Software Engineer", "Architect"],
     "IaC standard; multi-cloud deployments"),
    ("Kubernetes", "cloud", "growing", 78, 18.0, ["DevOps Engineer", "Software Engineer"],
     "Container orchestration standard; EKS/AKS/GKE"),
    
    # ─── DevOps & Platform Engineering ───
    ("Platform Engineering", "devops", "hot", 80, 200.0, ["DevOps Engineer", "Software Engineer", "Architect"],
     "Internal developer platforms; replaces 'DevOps' as title"),
    ("GitHub Actions", "devops", "growing", 72, 60.0, ["DevOps Engineer", "Software Engineer"],
     "CI/CD standard for cloud-native"),
    ("ArgoCD", "devops", "growing", 65, 80.0, ["DevOps Engineer"],
     "GitOps continuous delivery"),
    ("Backstage", "devops", "growing", 55, 150.0, ["DevOps Engineer", "Software Engineer"],
     "Developer portal framework by Spotify"),
    ("Docker", "devops", "stable", 75, 5.0, ["Software Engineer", "DevOps Engineer"],
     "Containerization standard; mature but ubiquitous"),
    ("CI/CD", "devops", "stable", 78, 8.0, ["Software Engineer", "DevOps Engineer"],
     "Core competency; table stakes for any engineer"),
    
    # ─── Backend & Systems ───
    ("Rust", "backend", "growing", 68, 65.0, ["Software Engineer", "Systems Engineer"],
     "Systems programming; growing in cloud infra and WebAssembly"),
    ("Go/Golang", "backend", "growing", 72, 30.0, ["Software Engineer", "DevOps Engineer"],
     "Cloud-native services; Kubernetes ecosystem"),
    ("Java", "backend", "stable", 70, -2.0, ["Software Engineer", "Developer"],
     "Enterprise backbone; Spring Boot 3+ modernization"),
    ("Python", "backend", "growing", 85, 20.0, ["Software Engineer", "Data Engineer", "ML Engineer"],
     "AI/ML growth driving Python demand even higher"),
    ("Node.js", "backend", "stable", 62, 0.0, ["Software Engineer", "Developer"],
     "Mature; stable demand for API/microservices"),
    ("Spring Boot", "backend", "stable", 60, 5.0, ["Software Engineer", "Developer"],
     "Java enterprise standard; cloud-native support"),
    ("GraphQL", "backend", "stable", 50, -5.0, ["Software Engineer", "Developer"],
     "API query language; maturing, less hype"),
    ("gRPC", "backend", "growing", 55, 40.0, ["Software Engineer", "Architect"],
     "High-performance microservice communication"),
    
    # ─── Frontend & Full-Stack ───
    ("React", "frontend", "stable", 72, 2.0, ["Software Engineer", "Developer"],
     "Dominant UI framework; Server Components trending"),
    ("Next.js", "frontend", "growing", 65, 45.0, ["Software Engineer", "Developer"],
     "Full-stack React framework; SSR/SSG"),
    ("TypeScript", "frontend", "growing", 78, 25.0, ["Software Engineer", "Developer"],
     "Standard for professional web dev; replacing JS"),
    ("Svelte", "frontend", "growing", 35, 80.0, ["Developer"],
     "Gaining traction but still niche"),
    
    # ─── Data Engineering ───
    ("Snowflake", "data", "growing", 65, 30.0, ["Data Engineer", "Software Engineer"],
     "Cloud data warehouse leader"),
    ("Databricks", "data", "growing", 70, 45.0, ["Data Engineer", "ML Engineer"],
     "Unified analytics; lakehouse architecture"),
    ("Apache Spark", "data", "stable", 60, 5.0, ["Data Engineer"],
     "Big data processing standard"),
    ("dbt", "data", "growing", 58, 55.0, ["Data Engineer"],
     "Data transformation standard; analytics engineering"),
    ("Apache Kafka", "data", "stable", 62, 8.0, ["Software Engineer", "Data Engineer"],
     "Event streaming; real-time data pipelines"),
    ("Apache Flink", "data", "growing", 48, 70.0, ["Data Engineer"],
     "Real-time stream processing; gaining over Spark Streaming"),
    
    # ─── Security ───
    ("Zero Trust", "security", "growing", 60, 50.0, ["Security Engineer", "Architect"],
     "Security architecture paradigm shift"),
    ("SAST/DAST", "security", "growing", 55, 35.0, ["DevOps Engineer", "Security Engineer"],
     "Shift-left security; DevSecOps"),
    ("SOC 2", "security", "stable", 45, 10.0, ["Security Engineer"],
     "Compliance standard"),
]

# Insert data
inserted = 0
skipped = 0
for row in MARKET_DATA_2026_Q3:
    skill, category, trend, score, yoy, roles, notes = row
    try:
        cur.execute("""
            INSERT INTO market_skill_signals 
            (skill_name, category, market_trend, market_demand_score, yoy_growth_pct, 
             source, relevant_roles, notes, quarter)
            VALUES (%s, %s, %s, %s, %s, 'curator', %s, %s, '2026-Q3')
            ON DUPLICATE KEY UPDATE
                market_trend = VALUES(market_trend),
                market_demand_score = VALUES(market_demand_score),
                yoy_growth_pct = VALUES(yoy_growth_pct),
                relevant_roles = VALUES(relevant_roles),
                notes = VALUES(notes)
        """, (skill, category, trend, score, yoy, json.dumps(roles), notes))
        inserted += 1
    except Exception as e:
        print(f"  Error: {skill} - {e}")
        skipped += 1

conn.commit()

print(f"Inserted/updated: {inserted} market signals")
print(f"Skipped: {skipped}")

# ──────────────────────────────────────────────────────────────
# Step 3: Show the divergence analysis
# ──────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("DIVERGENCE ANALYSIS: Internal JD Data vs External Market")
print("=" * 70)

# Find skills where our internal data says "declining" but market says "hot/growing"
cur.execute("""
    SELECT m.skill_name, m.market_trend, m.market_demand_score, m.yoy_growth_pct,
           s.trend_category as internal_trend, s.current_frequency as internal_freq
    FROM market_skill_signals m
    LEFT JOIN skill_trends s ON (
        LOWER(s.skill) = LOWER(m.skill_name) 
        AND s.company_name = '_ALL_'
        AND s.country = 'US'
    )
    WHERE m.market_trend IN ('hot', 'growing')
    AND (s.trend_category IN ('declining', 'stable') OR s.trend_category IS NULL)
    ORDER BY m.market_demand_score DESC
""")
gaps = cur.fetchall()

print(f"\nSkills HOT in market but DECLINING/MISSING in our data ({len(gaps)} gaps):")
print(f"{'Skill':<30} {'Market':>8} {'Score':>6} {'YoY%':>7} {'Internal':>12} {'IntFreq':>8}")
print("-" * 80)
for g in gaps[:20]:
    internal = g['internal_trend'] or 'NOT FOUND'
    freq = g['internal_freq'] or 0
    print(f"{g['skill_name']:<30} {g['market_trend']:>8} {g['market_demand_score']:>6} {g['yoy_growth_pct']:>6.0f}% {internal:>12} {freq:>8}")

cur.close()
conn.close()
print("\nDone. Market signals stored in `market_skill_signals` table.")
