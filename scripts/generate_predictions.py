"""
Generate job demand predictions using historical data + Azure OpenAI.

- Aggregates updated_job_records by quarter
- Computes Q1 2026 & Q2 2026 actuals
- Feeds trends to Azure OpenAI to predict Q3 2026
- Stores results in quarter_predictions table
"""
import os
import json
import mysql.connector
import requests
from dotenv import load_dotenv
from collections import Counter

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────
DB_CONFIG = {
    "host": os.getenv("MYSQL_HOST"),
    "port": int(os.getenv("MYSQL_PORT")),
    "database": os.getenv("MYSQL_DATABASE"),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
}

# Azure OpenAI Configuration
AZURE_ENDPOINT = "https://aiopsproject.openai.azure.com/"
AZURE_API_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_API_VERSION = "2024-08-01-preview"

if not AZURE_API_KEY:
    raise ValueError("AZURE_OPENAI_KEY is not set")

# Practice keywords (same mapping used for skills)
PRACTICE_KEYWORDS = {
    "Engineering": ["developer", "engineer", "software", "full stack", "fullstack", "backend", "frontend", "java", "python", ".net", "c#", "golang", "rust", "microservices", "api"],
    "Data & Analytics": ["data analyst", "data engineer", "data scientist", "analytics", "bi ", "business intelligence", "tableau", "power bi", "etl", "data warehouse", "sql developer"],
    "AI/ML": ["machine learning", "artificial intelligence", "ai ", "ml ", "deep learning", "nlp", "computer vision", "llm", "generative ai", "chatgpt"],
    "Cloud & Infrastructure": ["cloud", "aws", "azure", "gcp", "devops", "infrastructure", "terraform", "kubernetes", "docker", "sre", "site reliability", "platform engineer"],
    "Cybersecurity": ["security", "cybersecurity", "infosec", "soc ", "penetration", "vulnerability", "iam", "identity", "compliance"],
    "Project Management": ["project manager", "program manager", "scrum master", "agile coach", "delivery manager", "pmo"],
    "QA": ["qa ", "quality assurance", "test ", "tester", "automation test", "sdet", "quality engineer"],
    "UI/UX & Frontend": ["ui ", "ux ", "designer", "figma", "frontend", "react", "angular", "vue", "user experience", "user interface"],
    "ERP & Business Apps": ["sap", "oracle erp", "salesforce", "dynamics", "servicenow", "workday", "peoplesoft", "netsuite"],
    "Database & Storage": ["dba", "database admin", "postgres", "mysql", "mongodb", "redis", "sql server", "oracle db"],
    "Networking & Telecom": ["network", "cisco", "telecom", "routing", "switching", "wireless", "5g", "voip"],
    "Business Analysis": ["business analyst", "ba ", "requirements", "process improvement", "business process"],
    "Sales & Marketing": ["marketing", "sales", "seo", "content", "digital marketing", "brand", "campaign"],
    "Finance & Accounting": ["accountant", "finance", "audit", "tax", "financial analyst", "bookkeep", "controller"],
    "HR & Talent": ["recruiter", "hr ", "human resources", "talent acquisition", "payroll", "compensation"],
}


def get_quarter(date_str):
    """Convert date string to quarter label like '2026-Q1'."""
    if not date_str:
        return None
    try:
        from datetime import datetime
        dt = datetime.fromisoformat(date_str)
        q = (dt.month - 1) // 3 + 1
        return f"{dt.year}-Q{q}"
    except:
        return None


def map_title_to_practice(title):
    """Map a job title to a practice using keywords."""
    if not title:
        return "Other"
    t = title.lower()
    for practice, keywords in PRACTICE_KEYWORDS.items():
        for kw in keywords:
            if kw in t:
                return practice
    return "Other"


def aggregate_quarter_data(cur):
    """Get job counts per quarter, grouped by title, company, practice, division."""
    print("  Aggregating historical data...")

    cur.execute("""
        SELECT title, company_name, division_name, issue_date, job_status
        FROM updated_job_records
        WHERE issue_date IS NOT NULL
        ORDER BY issue_date
    """)
    rows = cur.fetchall()
    print(f"  Total records: {len(rows)}")

    # Group by quarter
    quarters = {}
    for title, company, division, issue_date, status in rows:
        q = get_quarter(str(issue_date))
        if not q:
            continue
        if q not in quarters:
            quarters[q] = {"titles": [], "companies": [], "practices": [], "divisions": [], "total": 0}
        quarters[q]["titles"].append(title or "Unknown")
        quarters[q]["companies"].append(company or "Unknown")
        quarters[q]["divisions"].append(division or "Unknown")
        quarters[q]["practices"].append(map_title_to_practice(title))
        quarters[q]["total"] += 1

    # Aggregate counts
    result = {}
    for q, data in sorted(quarters.items()):
        result[q] = {
            "total": data["total"],
            "titles": Counter(data["titles"]).most_common(15),
            "companies": Counter(data["companies"]).most_common(15),
            "practices": Counter(data["practices"]).most_common(15),
            "divisions": Counter(data["divisions"]).most_common(10),
        }
        print(f"    {q}: {data['total']} jobs")

    return result


def calculate_trends(quarter_data):
    """Calculate quarter-over-quarter trends."""
    sorted_quarters = sorted(quarter_data.keys())

    # We need at least Q1 2026 and Q2 2026
    trends = {}
    for i, q in enumerate(sorted_quarters):
        if i == 0:
            continue
        prev_q = sorted_quarters[i - 1]
        prev_total = quarter_data[prev_q]["total"]
        curr_total = quarter_data[q]["total"]
        pct = ((curr_total - prev_total) / prev_total * 100) if prev_total > 0 else 0
        trends[q] = {
            "total": curr_total,
            "prev_total": prev_total,
            "trend_pct": round(pct, 1),
        }

    return trends


def call_azure_openai(prompt):
    """Call Azure OpenAI for predictions."""
    # Try common deployment names
    deployments_to_try = ["gpt-4o", "gpt-4", "gpt-35-turbo", "gpt-4o-mini"]

    for deployment in deployments_to_try:
        url = f"{AZURE_ENDPOINT}openai/deployments/{deployment}/chat/completions?api-version={AZURE_API_VERSION}"
        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_API_KEY,
        }
        body = {
            "messages": [
                {"role": "system", "content": "You are an expert workforce analytics advisor. Analyze job market trends and provide predictions in JSON format."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.3,
            "max_tokens": 4000,
        }

        try:
            resp = requests.post(url, headers=headers, json=body, timeout=120)
            if resp.status_code == 200:
                print(f"  Using deployment: {deployment}")
                return resp.json()["choices"][0]["message"]["content"]
            elif resp.status_code == 404:
                continue  # try next deployment
            else:
                print(f"  {deployment}: {resp.status_code} - {resp.text[:200]}")
                continue
        except Exception as e:
            print(f"  {deployment} error: {e}")
            continue

    raise RuntimeError("No Azure OpenAI deployment found. Tried: " + str(deployments_to_try))


def generate_predictions_for_quarter(quarter_data, trends, target_quarter, prev_quarter):
    """Use Azure OpenAI to generate predictions for a given quarter."""
    print(f"\n  Calling Azure OpenAI for {target_quarter} predictions...")

    # Build context for the AI
    sorted_quarters = sorted(quarter_data.keys())
    history_summary = []
    for q in sorted_quarters:
        d = quarter_data[q]
        top_titles = [f"{t[0]} ({t[1]})" for t in d["titles"][:5]]
        top_companies = [f"{c[0]} ({c[1]})" for c in d["companies"][:5]]
        top_practices = [f"{p[0]} ({p[1]})" for p in d["practices"][:5]]
        history_summary.append(f"""
{q}: {d['total']} total jobs
  Top titles: {', '.join(top_titles)}
  Top companies: {', '.join(top_companies)}
  Top practices: {', '.join(top_practices)}""")

    prompt = f"""Based on this historical job data (quarterly), predict {target_quarter}.

HISTORICAL DATA:
{''.join(history_summary)}

TASK: Predict {target_quarter} job demand. Return ONLY valid JSON (no markdown, no code blocks) in this exact format:
{{
  "quarter": "{target_quarter}",
  "total_predicted_jobs": <number>,
  "ai_summary": "<2-3 sentence summary of the prediction>",
  "titles": [
    {{"name": "<job title>", "predicted_count": <number>, "trend_pct": <number>, "trend_direction": "up|down|stable", "confidence": "high|medium|low"}}
  ],
  "companies": [
    {{"name": "<company>", "predicted_count": <number>, "trend_pct": <number>, "trend_direction": "up|down|stable", "confidence": "high|medium|low"}}
  ],
  "practices": [
    {{"name": "<practice>", "predicted_count": <number>, "trend_pct": <number>, "trend_direction": "up|down|stable", "confidence": "high|medium|low"}}
  ]
}}

Give top 15 titles, top 15 companies, and all practices. Base predictions on quarterly trends.
Trend_pct should be relative to {prev_quarter}. Confidence based on consistency of the trend."""

    response = call_azure_openai(prompt)

    # Parse JSON from response (handle possible markdown wrapping)
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        text = text.rsplit("```", 1)[0]
    if text.startswith("json"):
        text = text[4:]

    # Try to fix common JSON issues (trailing commas, etc.)
    import re
    text = re.sub(r',\s*([}\]])', r'\1', text)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  JSON parse error: {e}")
        print(f"  Retrying API call...")
        response = call_azure_openai(prompt)
        text = response.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0]
        if text.startswith("json"):
            text = text[4:]
        text = re.sub(r',\s*([}\]])', r'\1', text)
        return json.loads(text)


def store_actuals(cur, quarter_data):
    """Store Q1 2026 and Q2 2026 actual data."""
    print("\n  Storing actual data for Q1 & Q2 2026...")

    for q in ["2026-Q1", "2026-Q2"]:
        if q not in quarter_data:
            print(f"    {q} not found in data, skipping")
            continue

        data = quarter_data[q]
        prev_q = "2025-Q4" if q == "2026-Q1" else "2026-Q1"
        prev_total = quarter_data.get(prev_q, {}).get("total", 0)

        # Store total
        total = data["total"]
        trend_pct = ((total - prev_total) / prev_total * 100) if prev_total > 0 else 0
        direction = "up" if trend_pct > 2 else ("down" if trend_pct < -2 else "stable")

        cur.execute("""INSERT INTO quarter_predictions
            (quarter, type, category, name, predicted_count, previous_count, trend_pct, trend_direction, confidence)
            VALUES (%s, 'actual', 'total', 'Total Jobs', %s, %s, %s, %s, 'actual')""",
            (q, total, prev_total, round(trend_pct, 2), direction))

        # Store top titles
        for name, count in data["titles"][:15]:
            cur.execute("""INSERT INTO quarter_predictions
                (quarter, type, category, name, predicted_count, trend_direction, confidence)
                VALUES (%s, 'actual', 'title', %s, %s, 'actual', 'actual')""",
                (q, name, count))

        # Store top companies
        for name, count in data["companies"][:15]:
            cur.execute("""INSERT INTO quarter_predictions
                (quarter, type, category, name, predicted_count, trend_direction, confidence)
                VALUES (%s, 'actual', 'company', %s, %s, 'actual', 'actual')""",
                (q, name, count))

        # Store practices
        for name, count in data["practices"][:15]:
            cur.execute("""INSERT INTO quarter_predictions
                (quarter, type, category, name, predicted_count, trend_direction, confidence)
                VALUES (%s, 'actual', 'practice', %s, %s, 'actual', 'actual')""",
                (q, name, count))

        print(f"    {q}: stored (total={total})")


def store_predictions(cur, predictions, quarter_data, target_quarter, prev_quarter):
    """Store AI predictions for a given quarter."""
    print(f"\n  Storing predictions for {target_quarter}...")

    q = target_quarter
    q2_total = quarter_data.get(prev_quarter, {}).get("total", 0)

    # Total
    total = predictions.get("total_predicted_jobs", 0)
    trend_pct = ((total - q2_total) / q2_total * 100) if q2_total > 0 else 0
    direction = "up" if trend_pct > 2 else ("down" if trend_pct < -2 else "stable")
    ai_summary = predictions.get("ai_summary", "")

    cur.execute("""INSERT INTO quarter_predictions
        (quarter, type, category, name, predicted_count, previous_count, trend_pct, trend_direction, confidence, ai_insight)
        VALUES (%s, 'predicted', 'total', 'Total Jobs', %s, %s, %s, %s, 'high', %s)""",
        (q, total, q2_total, round(trend_pct, 2), direction, ai_summary))

    # Titles
    for item in predictions.get("titles", []):
        cur.execute("""INSERT INTO quarter_predictions
            (quarter, type, category, name, predicted_count, trend_pct, trend_direction, confidence)
            VALUES (%s, 'predicted', 'title', %s, %s, %s, %s, %s)""",
            (q, item["name"], item["predicted_count"], item.get("trend_pct", 0),
             item.get("trend_direction", "stable"), item.get("confidence", "medium")))

    # Companies
    for item in predictions.get("companies", []):
        cur.execute("""INSERT INTO quarter_predictions
            (quarter, type, category, name, predicted_count, trend_pct, trend_direction, confidence)
            VALUES (%s, 'predicted', 'company', %s, %s, %s, %s, %s)""",
            (q, item["name"], item["predicted_count"], item.get("trend_pct", 0),
             item.get("trend_direction", "stable"), item.get("confidence", "medium")))

    # Practices
    for item in predictions.get("practices", []):
        cur.execute("""INSERT INTO quarter_predictions
            (quarter, type, category, name, predicted_count, trend_pct, trend_direction, confidence)
            VALUES (%s, 'predicted', 'practice', %s, %s, %s, %s, %s)""",
            (q, item["name"], item["predicted_count"], item.get("trend_pct", 0),
             item.get("trend_direction", "stable"), item.get("confidence", "medium")))

    print(f"    {target_quarter}: stored (predicted total={total})")


def main():
    print("=" * 60)
    print("JOB DEMAND PREDICTIONS")
    print("=" * 60)

    conn = mysql.connector.connect(**DB_CONFIG)
    cur = conn.cursor()

    # Clear previous predictions
    cur.execute("TRUNCATE TABLE quarter_predictions")
    conn.commit()

    # Step 1: Aggregate historical data
    quarter_data = aggregate_quarter_data(cur)

    # Step 2: Calculate trends
    trends = calculate_trends(quarter_data)

    # Step 3: Store actuals (Q1 & Q2 2026)
    store_actuals(cur, quarter_data)
    conn.commit()

    # Step 4: Generate AI predictions for Q3 2026
    predictions_q3 = generate_predictions_for_quarter(quarter_data, trends, "2026-Q3", "2026-Q2")

    # Step 5: Store Q3 predictions
    store_predictions(cur, predictions_q3, quarter_data, "2026-Q3", "2026-Q2")
    conn.commit()

    # Step 6: Generate AI predictions for Q4 2026
    # Add Q3 predictions as synthetic data so AI can use it for Q4
    quarter_data["2026-Q3"] = {
        "total": predictions_q3.get("total_predicted_jobs", 0),
        "titles": [(t["name"], t["predicted_count"]) for t in predictions_q3.get("titles", [])],
        "companies": [(c["name"], c["predicted_count"]) for c in predictions_q3.get("companies", [])],
        "practices": [(p["name"], p["predicted_count"]) for p in predictions_q3.get("practices", [])],
        "divisions": [],
    }
    predictions_q4 = generate_predictions_for_quarter(quarter_data, trends, "2026-Q4", "2026-Q3")

    # Step 7: Store Q4 predictions
    store_predictions(cur, predictions_q4, quarter_data, "2026-Q4", "2026-Q3")
    conn.commit()

    # Summary
    cur.execute("SELECT COUNT(*) FROM quarter_predictions")
    total_rows = cur.fetchone()[0]

    cur.close()
    conn.close()

    print(f"\n{'='*60}")
    print(f"DONE! {total_rows} rows in quarter_predictions table")
    print(f"  - Q1 2026: actual")
    print(f"  - Q2 2026: actual")
    print(f"  - Q3 2026: predicted (AI)")
    print(f"  - Q4 2026: predicted (AI)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
