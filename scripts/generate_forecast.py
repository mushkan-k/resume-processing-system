"""
Generate Job Forecast — predicts future roles with requirements profiles.

For each predicted role (e.g. "Software Engineer III"), it generates:
- Predicted count for the quarter
- Required skills & experience
- Typical locations, companies, bill rates
- AI summary of what HR should prepare for

Steps:
1. Aggregate historical jobs by role title per quarter
2. Identify top roles by volume
3. For each top role, analyze historical requirements (skills field)
4. Call Azure OpenAI to predict next quarters with full requirement profiles
5. Store in job_forecast + job_forecast_roles tables
"""
import os
import json
import re
import mysql.connector
import requests
from dotenv import load_dotenv
from collections import Counter
from datetime import datetime

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────
DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", "3305")),
    "database": os.getenv("MYSQL_DATABASE", "resume_processing"),
    "user": os.getenv("MYSQL_USER", "resume_user"),
    "password": os.getenv("MYSQL_PASSWORD", "resume_password"),
}

# Azure OpenAI Configuration
AZURE_ENDPOINT = "https://aiopsproject.openai.azure.com/"
AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_API_VERSION = "2024-08-01-preview"

if not AZURE_API_KEY:
    raise ValueError("AZURE_OPENAI_KEY is not set")

PRACTICE_KEYWORDS = {
    "Engineering": ["developer", "engineer", "software", "full stack", "fullstack", "backend", "frontend", "java", "python", ".net", "c#", "golang", "rust", "microservices", "api"],
    "Data & Analytics": ["data analyst", "data engineer", "data scientist", "analytics", "bi ", "business intelligence", "tableau", "power bi", "etl", "data warehouse", "sql developer"],
    "AI/ML": ["machine learning", "artificial intelligence", "ai ", "ml ", "deep learning", "nlp", "computer vision", "llm", "generative ai"],
    "Cloud & Infrastructure": ["cloud", "aws", "azure", "gcp", "devops", "infrastructure", "terraform", "kubernetes", "docker", "sre", "site reliability", "platform engineer"],
    "Cybersecurity": ["security", "cybersecurity", "infosec", "soc ", "penetration", "vulnerability", "iam", "identity"],
    "Project Management": ["project manager", "program manager", "scrum master", "agile coach", "delivery manager", "pmo"],
    "QA": ["qa ", "quality assurance", "test ", "tester", "automation test", "sdet", "quality engineer"],
    "UI/UX & Frontend": ["ui ", "ux ", "designer", "figma", "frontend", "react", "angular", "vue", "user experience"],
    "ERP & Business Apps": ["sap", "oracle erp", "salesforce", "dynamics", "servicenow", "workday", "peoplesoft"],
    "Database & Storage": ["dba", "database admin", "postgres", "mysql", "mongodb", "redis", "sql server"],
    "Networking & Telecom": ["network", "cisco", "telecom", "routing", "switching", "wireless", "5g"],
    "Business Analysis": ["business analyst", "ba ", "requirements", "process improvement"],
    "Sales & Marketing": ["marketing", "sales", "seo", "content", "digital marketing"],
    "Finance & Accounting": ["accountant", "finance", "audit", "tax", "financial analyst"],
    "HR & Talent": ["recruiter", "hr ", "human resources", "talent acquisition", "payroll"],
}


def get_quarter(date_val):
    """Convert date to quarter label."""
    if not date_val:
        return None
    try:
        if isinstance(date_val, str):
            date_val = datetime.fromisoformat(date_val)
        q = (date_val.month - 1) // 3 + 1
        return f"{date_val.year}-Q{q}"
    except:
        return None


def map_to_practice(title):
    if not title:
        return "Other"
    t = title.lower()
    for practice, keywords in PRACTICE_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                return practice
    return "Other"


def call_azure_openai(prompt, max_tokens=4000):
    """Call Azure OpenAI."""
    deployments = ["gpt-4o-mini", "gpt-4o", "gpt-4"]
    for deployment in deployments:
        url = f"{AZURE_ENDPOINT}openai/deployments/{deployment}/chat/completions?api-version={AZURE_API_VERSION}"
        headers = {"Content-Type": "application/json", "api-key": AZURE_API_KEY}
        body = {
            "messages": [
                {"role": "system", "content": "You are an expert workforce analytics advisor. Provide predictions in valid JSON only. No markdown."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": max_tokens,
        }
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=180)
            if resp.status_code == 200:
                print(f"    [OK] deployment: {deployment}")
                return resp.json()["choices"][0]["message"]["content"]
            elif resp.status_code == 404:
                continue
            else:
                print(f"    [{resp.status_code}] {deployment}: {resp.text[:100]}")
                continue
        except Exception as e:
            print(f"    [ERR] {deployment}: {e}")
            continue
    raise RuntimeError("No Azure OpenAI deployment available")


def parse_json_response(text):
    """Parse JSON from AI response, handling markdown wrapping."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    if text.startswith("json"):
        text = text[4:]
    # Fix trailing commas
    text = re.sub(r',\s*([}\]])', r'\1', text)
    return json.loads(text)


def aggregate_data(cur):
    """Get all jobs grouped by quarter and role."""
    print("\n[1] Aggregating historical data...")
    cur.execute("""
        SELECT title, company_name, city, state, country, issue_date,
               bill_rate_min, bill_rate_max, skills, job_status
        FROM updated_job_records
        WHERE issue_date IS NOT NULL AND title IS NOT NULL
        ORDER BY issue_date
    """)
    rows = cur.fetchall()
    print(f"    Total records: {len(rows)}")

    # Group by quarter
    quarters = {}
    # Also build per-role profiles
    role_profiles = {}

    for title, company, city, state, country, issue_date, br_min, br_max, skills, status in rows:
        q = get_quarter(issue_date)
        if not q:
            continue

        # Quarter totals
        if q not in quarters:
            quarters[q] = {"total": 0, "roles": Counter(), "companies": Counter()}
        quarters[q]["total"] += 1
        quarters[q]["roles"][title] += 1
        quarters[q]["companies"][company or "Unknown"] += 1

        # Role profile (aggregate across all time)
        if title not in role_profiles:
            role_profiles[title] = {
                "skills_raw": [],
                "companies": [],
                "locations": [],
                "quarters": Counter(),
            }
        if skills:
            role_profiles[title]["skills_raw"].append(skills)
        if company:
            role_profiles[title]["companies"].append(company)
        loc = f"{city}, {state}" if city and state else (city or state or "")
        if loc:
            role_profiles[title]["locations"].append(loc)
        role_profiles[title]["quarters"][q] += 1

    for q in sorted(quarters.keys()):
        print(f"    {q}: {quarters[q]['total']} jobs, {len(quarters[q]['roles'])} unique titles")

    return quarters, role_profiles


def extract_skills_from_raw(skills_list):
    """Parse the skills field into individual skill names."""
    all_skills = []
    for raw in skills_list[:50]:  # sample up to 50
        # Skills format: "(SKILL1 over X year(s) OR SKILL2 over X year(s)) AND ..."
        # Extract capitalized skill names
        found = re.findall(r'([A-Z][A-Z0-9/\.\+\#\- ]{1,40}?)(?:\s+over\s+|\s*\)|\s*$)', raw)
        all_skills.extend([s.strip() for s in found if len(s.strip()) > 1])
    return Counter(all_skills).most_common(15)


def build_role_summary(title, profile):
    """Build a summary for a role from its historical data."""
    top_skills = extract_skills_from_raw(profile["skills_raw"])
    top_companies = Counter(profile["companies"]).most_common(5)
    top_locations = Counter(profile["locations"]).most_common(5)

    # Determine experience from skills text
    exp_years = []
    for raw in profile["skills_raw"][:20]:
        years = re.findall(r'over\s+(\d+)\s+year', raw)
        exp_years.extend([int(y) for y in years])
    avg_exp = f"{sum(exp_years)/len(exp_years):.0f}+ years" if exp_years else "Not specified"

    return {
        "top_skills": [s[0] for s in top_skills],
        "top_companies": [c[0] for c in top_companies],
        "top_locations": [l[0] for l in top_locations],
        "experience_level": avg_exp,
        "total_historical_jobs": sum(profile["quarters"].values()),
        "quarterly_counts": dict(profile["quarters"]),
    }


def generate_forecast(quarters, role_profiles):
    """Call AI to predict Q3 & Q4 2026 with requirement profiles."""
    print("\n[2] Generating forecast with Azure OpenAI...")

    # Get top 20 roles by total volume in recent quarters (2025-Q3 onwards)
    recent_roles = Counter()
    for q in ["2025-Q3", "2025-Q4", "2026-Q1", "2026-Q2"]:
        if q in quarters:
            for role, count in quarters[q]["roles"].items():
                recent_roles[role] += count

    top_roles = recent_roles.most_common(20)
    print(f"    Top 20 roles (by recent volume): {[r[0][:30] for r in top_roles[:5]]}...")

    # Build role context with historical requirements
    role_contexts = []
    for role_title, total_count in top_roles:
        if role_title in role_profiles:
            summary = build_role_summary(role_title, role_profiles[role_title])
            quarterly = {q: summary["quarterly_counts"].get(q, 0)
                        for q in sorted(quarters.keys()) if q >= "2025-Q1"}
            role_contexts.append({
                "title": role_title,
                "practice": map_to_practice(role_title),
                "recent_total": total_count,
                "quarterly_trend": quarterly,
                "historical_skills": summary["top_skills"][:10],
                "experience": summary["experience_level"],
                "top_companies": summary["top_companies"][:3],
                "top_locations": summary["top_locations"][:3],
            })

    # Quarter totals for context
    q_totals = {q: quarters[q]["total"] for q in sorted(quarters.keys())}

    prompt = f"""You are a workforce analytics expert. Based on historical job data, predict job demand for Q3 2026 (Jul-Sep) and Q4 2026 (Oct-Dec).

QUARTERLY TOTALS:
{json.dumps(q_totals, indent=2)}

TOP 20 ROLES WITH HISTORICAL CONTEXT:
{json.dumps(role_contexts, indent=2)}

TASK: For EACH of the 20 roles above, predict Q3 and Q4 2026. Return ONLY valid JSON:
{{
  "quarters": {{
    "2026-Q3": {{
      "total_jobs": <predicted total across all roles>,
      "ai_summary": "<2-3 sentence summary for HR team about Q3 outlook. Use simple language.>",
      "trend_pct": <% change compared to SAME quarter last year i.e. Q3 2025>,
      "trend_direction": "growing|declining|stable"
    }},
    "2026-Q4": {{
      "total_jobs": <predicted total>,
      "ai_summary": "<2-3 sentence summary for HR team about Q4 outlook. Use simple language.>",
      "trend_pct": <% change compared to SAME quarter last year i.e. Q4 2025>,
      "trend_direction": "growing|declining|stable"
    }}
  }},
  "roles": [
    {{
      "title": "<exact title from input>",
      "practice": "<practice>",
      "q3_count": <number>,
      "q4_count": <number>,
      "q3_vs_last_year_pct": <% change vs 2025 average quarterly count for this role>,
      "q4_vs_last_year_pct": <% change vs 2025 average quarterly count for this role>,
      "trend_direction": "growing|declining|stable",
      "confidence": "high|medium|low",
      "required_skills": ["skill1", "skill2", ...up to 8 skills],
      "experience_level": "X+ years",
      "expected_companies": ["company1", "company2"],
      "expected_locations": ["city1", "city2"],
      "requirements_summary": "<1-2 sentence actionable advice for HR: what skills to look for, what type of candidates to prepare>"
    }}
  ]
}}

IMPORTANT:
- Compare each role's predicted count to the AVERAGE quarterly count from all of 2025 (full year baseline). This gives a stable, meaningful trend.
- trend_direction: "growing" means more demand than 2025 average, "declining" means less, "stable" means similar
- confidence: "high" = very consistent pattern, "medium" = some variation, "low" = unpredictable
- required_skills should reflect what employers will ask for based on historical patterns
- requirements_summary should be simple, actionable advice for recruiters
- Be realistic with counts based on the actual trends shown"""

    response = call_azure_openai(prompt, max_tokens=6000)
    return parse_json_response(response)


def store_forecast(cur, forecast, quarters):
    """Store forecast data."""
    print("\n[3] Storing forecast data...")

    # Store actuals for Q1 & Q2 2026 (vs 2025 yearly average)
    # Calculate 2025 average per quarter
    year_2025_total = sum(quarters.get(f"2025-Q{q}", {}).get("total", 0) for q in range(1, 5))
    year_2025_avg = year_2025_total / 4 if year_2025_total > 0 else 1

    for q, last_year_q in [("2026-Q1", "2025-Q1"), ("2026-Q2", "2025-Q2")]:
        if q not in quarters:
            continue
        total = quarters[q]["total"]
        trend = ((total - year_2025_avg) / year_2025_avg * 100)
        direction = "growing" if trend > 2 else ("declining" if trend < -2 else "stable")
        cur.execute("""
            INSERT INTO job_forecast (quarter, type, total_jobs, previous_quarter_jobs, trend_pct, trend_direction)
            VALUES (%s, 'actual', %s, %s, %s, %s)
        """, (q, total, round(year_2025_avg), round(trend, 2), direction))
        print(f"    {q} actual: {total} jobs (vs 2025 avg/qtr: {round(year_2025_avg)})")

    # Store predicted quarters (vs 2025 yearly average)
    for q_key, last_year_q in [("2026-Q3", "2025-Q3"), ("2026-Q4", "2025-Q4")]:
        qdata = forecast["quarters"].get(q_key, {})
        total = qdata.get("total_jobs", 0)
        trend = ((total - year_2025_avg) / year_2025_avg * 100)
        direction = "growing" if trend > 2 else ("declining" if trend < -2 else "stable")
        cur.execute("""
            INSERT INTO job_forecast (quarter, type, total_jobs, previous_quarter_jobs, trend_pct, trend_direction, ai_summary)
            VALUES (%s, 'predicted', %s, %s, %s, %s, %s)
        """, (q_key, total, round(year_2025_avg), round(trend, 2), direction, qdata.get("ai_summary", "")))
        print(f"    {q_key} predicted: {total} jobs (vs 2025 avg/qtr: {round(year_2025_avg)})")

    # Store role predictions
    for role in forecast.get("roles", []):
        # Calculate 2025 yearly average for this role
        role_2025_total = sum(
            quarters.get(f"2025-Q{q}", {}).get("roles", Counter()).get(role["title"], 0)
            for q in range(1, 5)
        )
        role_2025_avg = role_2025_total / 4

        for q_key, count_key, yoy_key in [("2026-Q3", "q3_count", "q3_vs_last_year_pct"), ("2026-Q4", "q4_count", "q4_vs_last_year_pct")]:
            count = role.get(count_key, 0)
            prev_count = round(role_2025_avg)
            trend = ((count - role_2025_avg) / role_2025_avg * 100) if role_2025_avg > 0 else 0

            cur.execute("""
                INSERT INTO job_forecast_roles
                (quarter, role_title, predicted_count, previous_count, trend_pct, trend_direction,
                 confidence, practice, top_skills, experience_level, typical_locations,
                 typical_companies, bill_rate_range, ai_requirements_summary)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                q_key,
                role["title"],
                count,
                prev_count,
                round(trend, 2),
                "growing" if trend > 5 else ("declining" if trend < -5 else "stable"),
                role.get("confidence", "medium"),
                role.get("practice", "Other"),
                json.dumps(role.get("required_skills", [])),
                role.get("experience_level", ""),
                json.dumps(role.get("expected_locations", [])),
                json.dumps(role.get("expected_companies", [])),
                "",
                role.get("requirements_summary", ""),
            ))

    print(f"    Stored {len(forecast.get('roles', []))} roles x 2 quarters")

    # Also store actuals for Q1 & Q2 roles (top 15)
    for q in ["2026-Q1", "2026-Q2"]:
        if q not in quarters:
            continue
        for title, count in quarters[q]["roles"].most_common(15):
            # Compare to 2025 average for this role
            role_2025_total = sum(
                quarters.get(f"2025-Q{qi}", {}).get("roles", Counter()).get(title, 0)
                for qi in range(1, 5)
            )
            role_2025_avg = role_2025_total / 4
            prev_count = round(role_2025_avg)
            trend = ((count - role_2025_avg) / role_2025_avg * 100) if role_2025_avg > 0 else 0
            # Get historical profile for requirements
            profile = role_profiles.get(title)
            skills = []
            exp = ""
            locs = []
            companies = []
            if profile:
                summary = build_role_summary(title, profile)
                skills = summary["top_skills"][:10]
                exp = summary["experience_level"]
                locs = summary["top_locations"][:5]
                companies = summary["top_companies"][:5]

            cur.execute("""
                INSERT INTO job_forecast_roles
                (quarter, role_title, predicted_count, previous_count, trend_pct, trend_direction,
                 confidence, practice, top_skills, experience_level, typical_locations,
                 typical_companies, bill_rate_range, ai_requirements_summary)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                q, title, count, prev_count, round(trend, 2),
                "growing" if trend > 5 else ("declining" if trend < -5 else "stable"),
                "actual", map_to_practice(title),
                json.dumps(skills), exp, json.dumps(locs), json.dumps(companies),
                "", f"Based on {count} actual positions in {q}",
            ))


# We need role_profiles accessible in store_forecast
role_profiles = {}


def main():
    global role_profiles
    print("=" * 60)
    print("JOB FORECAST — Role Predictions with Requirements")
    print("=" * 60)

    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Clear old data
    cur.execute("TRUNCATE TABLE job_forecast")
    cur.execute("TRUNCATE TABLE job_forecast_roles")
    conn.commit()

    # Step 1: Aggregate
    quarters, role_profiles_local = aggregate_data(cur)
    role_profiles = role_profiles_local

    # Step 2 & 3: Generate AI predictions
    forecast = generate_forecast(quarters, role_profiles)

    # Step 4: Store
    store_forecast(cur, forecast, quarters)
    conn.commit()

    # Summary
    cur.execute("SELECT COUNT(*) FROM job_forecast")
    f_count = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM job_forecast_roles")
    r_count = cur.fetchone()[0]

    cur.close()
    conn.close()

    print(f"\n{'='*60}")
    print(f"DONE!")
    print(f"  job_forecast: {f_count} quarters")
    print(f"  job_forecast_roles: {r_count} role predictions")
    print(f"  Quarters: Q1 (actual), Q2 (actual), Q3 (predicted), Q4 (predicted)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
