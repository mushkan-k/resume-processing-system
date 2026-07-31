"""
Skills Forecasting Model
========================
Predicts which skills will be needed per cluster for future quarters.

Approach:
1. Build training data from jd_extracted_skills across ALL quarters
2. Classify skills as:
   - Foundation (appear in 80%+ of quarters for a cluster) 
   - Trending Up (frequency increasing)
   - Trending Down (frequency decreasing)
   - Emerging (new in recent quarters)
3. For predicted quarters (Q3/Q4), output:
   - Foundation skills (always included)
   - Trending/emerging skills (weighted by momentum)
   - Drop declining skills

Output: predicted_cluster_skills table
"""
import os, sys, json, re, math
from datetime import datetime
from collections import defaultdict
import mysql.connector
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))
DB_CFG = dict(host='localhost', port=3305, user='root', password='rootpassword', database='resume_processing')

SKIP_SKILLS = {
    "develop", "design", "web", "work", "team", "support",
    "manage", "build", "create", "test", "project", "system",
    "data", "application", "software", "service", "experience",
    "years", "null", "linkedin", "engineer", "engineering",
    "samsung", "nokia", "cisco", "ericsson", "fujitsu", "corning",
    "client", "communication", "skills", "ability", "responsible",
    "knowledge", "working", "management", "required", "preferred",
    "minimum", "bachelor", "degree", "strong", "excellent",
    "commscope", "qualcomm", "motorola", "huawei", "juniper",
    "t-mobile", "at&t", "verizon", "sprint", "comcast",
    "accenture", "infosys", "tcs", "wipro", "cognizant",
    "apple", "google", "microsoft", "amazon", "meta", "ibm",
}


def _is_valid_skill(skill, min_strength=0.01):
    return (
        skill
        and skill.lower() not in SKIP_SKILLS
        and len(skill) > 2
    )


_onet_cache = None

def _load_onet_taxonomy():
    """Load O*NET skills taxonomy for market signal enrichment."""
    global _onet_cache
    if _onet_cache is not None:
        return _onet_cache
    onet_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'market_signals', 'onet_skills_taxonomy.json')
    try:
        with open(onet_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        _onet_cache = data.get('cluster_skills', {})
        print("   Loaded O*NET taxonomy: %d occupations" % len(_onet_cache))
    except Exception as e:
        print("   Warning: Could not load O*NET taxonomy: %s" % e)
        _onet_cache = {}
    return _onet_cache


def build_training_data(cur):
    """Pull all skill data across quarters and build per-cluster skill timelines."""
    cur.execute("""
        SELECT cluster_name, quarter, title, extracted_skills
        FROM jd_extracted_skills
        ORDER BY cluster_name, quarter
    """)
    rows = cur.fetchall()
    print("Total jd_extracted_skills records: %d" % len(rows))

    # Structure: cluster -> quarter -> skill -> count
    cluster_quarter_skills = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    # Also track: cluster -> quarter -> total_jds (for normalization)
    cluster_quarter_jds = defaultdict(lambda: defaultdict(int))

    for row in rows:
        cluster = row['cluster_name']
        quarter = row['quarter']
        try:
            skills = json.loads(row['extracted_skills'])
        except:
            continue

        cluster_quarter_jds[cluster][quarter] += 1
        for i, skill in enumerate(skills):
            # Equal weight - frequency matters more than position
            cluster_quarter_skills[cluster][quarter][skill] += 1

    return cluster_quarter_skills, cluster_quarter_jds


def classify_skills(cluster_quarter_skills, cluster_quarter_jds):
    """
    For each cluster, classify each skill:
    - foundation: appears in 70%+ of quarters with consistent frequency
    - trending_up: frequency increasing over recent quarters
    - trending_down: frequency decreasing
    - emerging: only in most recent quarter(s)
    - stable: appears regularly but not foundation-level
    """
    cluster_skill_forecasts = {}

    for cluster, quarter_skills in cluster_quarter_skills.items():
        quarters_sorted = sorted(quarter_skills.keys())  # e.g. ['2025Q3', '2025Q4', '2026Q1', '2026Q2']
        n_quarters = len(quarters_sorted)

        if n_quarters == 0:
            continue

        # Collect all skills seen in this cluster
        all_skills = set()
        for q_skills in quarter_skills.values():
            all_skills.update(q_skills.keys())

        skill_classifications = {}

        for skill in all_skills:
            # Build timeline: normalized frequency per quarter
            timeline = []
            presence_count = 0
            for q in quarters_sorted:
                jd_count = cluster_quarter_jds[cluster][q]
                if jd_count == 0:
                    timeline.append(0)
                    continue
                freq = quarter_skills[q].get(skill, 0) / jd_count
                timeline.append(freq)
                if quarter_skills[q].get(skill, 0) > 0:
                    presence_count += 1

            presence_ratio = presence_count / n_quarters if n_quarters > 0 else 0

            # Classification logic
            if presence_ratio >= 0.6 and n_quarters >= 2:
                category = "foundation"
            elif n_quarters == 1:
                # Only one quarter: treat all skills as foundation (best we have)
                category = "foundation"
            elif n_quarters >= 2 and presence_count == 1 and timeline[-1] > 0:
                category = "emerging"
            elif n_quarters >= 2:
                # Check trend: compare first half vs second half
                mid = n_quarters // 2
                first_half_avg = sum(timeline[:mid]) / max(mid, 1)
                second_half_avg = sum(timeline[mid:]) / max(n_quarters - mid, 1)

                if second_half_avg > first_half_avg * 1.3:
                    category = "trending_up"
                elif second_half_avg < first_half_avg * 0.5 and second_half_avg < 0.5:
                    category = "trending_down"
                else:
                    category = "stable"
            else:
                category = "stable"

            # Compute predicted strength for next quarter
            # Foundation: use average frequency
            # Trending: extrapolate
            # Emerging: boost
            if category == "foundation":
                predicted_strength = sum(timeline) / len(timeline) * 1.0
            elif category == "trending_up":
                predicted_strength = timeline[-1] * 1.2  # momentum boost
            elif category == "emerging":
                predicted_strength = timeline[-1] * 1.5  # new skill boost
            elif category == "trending_down":
                predicted_strength = timeline[-1] * 0.5  # decay
            else:
                predicted_strength = sum(timeline[-2:]) / max(len(timeline[-2:]), 1)

            skill_classifications[skill] = {
                "category": category,
                "presence_ratio": round(presence_ratio, 2),
                "predicted_strength": round(predicted_strength, 3),
                "last_freq": round(timeline[-1], 3) if timeline else 0,
                "timeline": [round(t, 2) for t in timeline],
            }

        cluster_skill_forecasts[cluster] = skill_classifications

    return cluster_skill_forecasts


def build_historical_breakdown(cluster, skill_data, cluster_quarter_skills, cluster_quarter_jds):
    """Historical JD skills grouped for recruiter planning."""
    quarters = sorted(cluster_quarter_skills.get(cluster, {}).keys())
    last_q = quarters[-1] if quarters else None

    recent_top = []
    if last_q:
        jd_count = cluster_quarter_jds[cluster].get(last_q, 0) or 1
        freq = cluster_quarter_skills[cluster][last_q]
        recent_top = [
            s for s, _ in sorted(freq.items(), key=lambda x: -x[1])
            if _is_valid_skill(s)
        ][:8]

    ranked = sorted(skill_data.items(), key=lambda x: -x[1]['predicted_strength'])
    filtered = [
        (s, d) for s, d in ranked
        if _is_valid_skill(s) and d['predicted_strength'] > 0.01
    ]

    return {
        "core": [s for s, d in filtered if d['category'] == 'foundation'][:8],
        "recent": recent_top[:8],
        "momentum": [s for s, d in filtered if d['category'] in ('trending_up', 'emerging')][:6],
        "stable": [s for s, d in filtered if d['category'] == 'stable'][:6],
        "source_quarters": quarters[-4:] if len(quarters) >= 4 else quarters,
        "quarters_count": len(quarters),
    }


def build_market_breakdown(cur, cluster_name):
    """External market signals mapped to this cluster's role.
    Sources: market_skill_signals DB table + O*NET tech_skills taxonomy.
    """
    plain_role = cluster_name.split(" | ", 1)[-1]
    
    # Source 1: DB market_skill_signals
    cur.execute("""
        SELECT skill_name, market_trend, market_demand_score, category, relevant_roles
        FROM market_skill_signals
        ORDER BY market_demand_score DESC
    """)
    rows = cur.fetchall()

    matched = []
    for row in rows:
        try:
            roles = json.loads(row['relevant_roles']) if row['relevant_roles'] else []
        except Exception:
            roles = []
        role_hit = any(
            plain_role.lower() in r.lower()
            or r.lower() in plain_role.lower()
            or any(word in plain_role.lower() for word in r.lower().split() if len(word) > 4)
            for r in roles
        )
        if role_hit or not roles:
            matched.append(row)

    if not matched:
        matched = rows[:30]

    hot, growing, stable = [], [], []
    for row in matched:
        skill = row['skill_name']
        if not _is_valid_skill(skill, min_strength=0):
            continue
        trend = (row.get('market_trend') or 'stable').lower()
        entry = {
            "skill": skill,
            "score": int(row.get('market_demand_score') or 0),
            "category": row.get('category') or 'general',
            "source": "market_signals",
        }
        if trend == 'hot':
            hot.append(entry)
        elif trend == 'growing':
            growing.append(entry)
        elif trend != 'declining':  # skip declining
            stable.append(entry)

    # Source 2: O*NET tech_skills taxonomy
    onet = _load_onet_taxonomy()
    # Match cluster role to O*NET occupation keys
    onet_skills = []
    onet_match_key = None
    for onet_key in onet:
        if onet_key.lower() == plain_role.lower():
            onet_match_key = onet_key
            break
        # Fuzzy: check if key words overlap
        onet_words = set(onet_key.lower().split())
        role_words = set(plain_role.lower().replace('-', ' ').split())
        if len(onet_words & role_words) >= 1 and len(onet_words) <= 3:
            onet_match_key = onet_key
            # Don't break — prefer exact match

    if onet_match_key:
        raw_onet = onet[onet_match_key]
        # tech_skills is the most useful list
        tech_list = raw_onet.get('tech_skills', [])
        # Filter to recognizable, short skill names (not full product names)
        for sk in tech_list:
            # Skip overly long O*NET entries like "Amazon Web Services AWS CloudFormation"
            # Keep concise ones like "Python", "Apache Kafka", "Kubernetes"
            if len(sk) <= 25 and _is_valid_skill(sk, min_strength=0):
                onet_skills.append(sk)

        # Deduplicate against DB skills already captured
        existing_lower = {e['skill'].lower() for e in hot + growing + stable}
        onet_unique = []
        seen_onet = set()
        for sk in onet_skills:
            if sk.lower() not in existing_lower and sk.lower() not in seen_onet:
                onet_unique.append({
                    "skill": sk,
                    "score": 50,  # default market relevance score for O*NET
                    "category": "O*NET taxonomy",
                    "source": "onet",
                })
                seen_onet.add(sk.lower())

        # Add top O*NET skills to growing (they represent industry-standard skills)
        growing.extend(onet_unique[:8])

    return {
        "hot": hot[:6],
        "growing": growing[:8],
        "stable": stable[:6],
        "onet_match": onet_match_key,  # track which O*NET occupation matched
    }


def generate_predictions(cur, cluster_skill_forecasts, cluster_quarter_skills, cluster_quarter_jds, target_quarters):
    """
    Generate predicted skills for each cluster for target quarters.
    Returns: list of (cluster, quarter, title_group, predicted_skills_json, category_breakdown)
    """
    predictions = []

    for cluster, skill_data in cluster_skill_forecasts.items():
        # Sort skills by predicted_strength
        ranked = sorted(skill_data.items(), key=lambda x: -x[1]['predicted_strength'])

        filtered = [
            (s, d) for s, d in ranked
            if _is_valid_skill(s)
            and d['predicted_strength'] > 0.01
            and d['category'] != 'trending_down'
        ]

        # Build predicted skills list
        foundation_skills = [s for s, d in filtered if d['category'] == 'foundation'][:10]
        trending_skills = [s for s, d in filtered if d['category'] == 'trending_up'][:5]
        emerging_skills = [s for s, d in filtered if d['category'] == 'emerging'][:3]
        stable_skills = [s for s, d in filtered if d['category'] == 'stable'][:5]

        # Final predicted set: foundation first, then trending, then emerging
        predicted = []
        seen = set()
        for s in foundation_skills + trending_skills + emerging_skills + stable_skills:
            if s.lower() not in seen:
                predicted.append(s)
                seen.add(s.lower())
            if len(predicted) >= 12:
                break

        predicted_breakdown = {
            "foundation": foundation_skills[:8],
            "trending_up": trending_skills[:5],
            "emerging": emerging_skills[:3],
            "stable": stable_skills[:5],
        }
        historical_breakdown = build_historical_breakdown(
            cluster, skill_data, cluster_quarter_skills, cluster_quarter_jds
        )
        market_breakdown = build_market_breakdown(cur, cluster)

        skills_intelligence = {
            "predicted": predicted_breakdown,
            "historical": historical_breakdown,
            "market": market_breakdown,
        }

        for q in target_quarters:
            predictions.append({
                "cluster_name": cluster,
                "quarter": q,
                "predicted_skills": predicted,
                "skill_categories": skills_intelligence,
                "confidence": "high" if len(foundation_skills) >= 3 else "medium",
                "model_version": "SkillForecast_v3",
            })

    return predictions


def save_predictions(cur, conn, predictions):
    """Save to predicted_cluster_skills table."""
    # Create table if not exists
    cur.execute("""
        CREATE TABLE IF NOT EXISTS predicted_cluster_skills (
            id INT AUTO_INCREMENT PRIMARY KEY,
            cluster_name VARCHAR(100) NOT NULL,
            quarter VARCHAR(10) NOT NULL,
            predicted_skills JSON NOT NULL,
            skill_categories JSON,
            confidence VARCHAR(20),
            model_version VARCHAR(50),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE KEY uq_cluster_quarter (cluster_name, quarter)
        )
    """)
    conn.commit()

    inserted = 0
    for pred in predictions:
        cur.execute("""
            INSERT INTO predicted_cluster_skills
            (cluster_name, quarter, predicted_skills, skill_categories, confidence, model_version)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                predicted_skills = VALUES(predicted_skills),
                skill_categories = VALUES(skill_categories),
                confidence = VALUES(confidence),
                model_version = VALUES(model_version),
                created_at = CURRENT_TIMESTAMP
        """, (
            pred['cluster_name'], pred['quarter'],
            json.dumps(pred['predicted_skills']),
            json.dumps(pred['skill_categories']),
            pred['confidence'], pred['model_version']
        ))
        inserted += 1

    conn.commit()
    return inserted


def main():
    conn = mysql.connector.connect(**DB_CFG)
    cur = conn.cursor(dictionary=True)

    print("=" * 60)
    print("SKILLS FORECASTING MODEL v3")
    print("=" * 60)

    # Step 1: Build training data
    print("\n[1] Building training data from all quarters...")
    cluster_quarter_skills, cluster_quarter_jds = build_training_data(cur)
    print("   Clusters with data: %d" % len(cluster_quarter_skills))
    print("   Quarters found: %s" % sorted(set(q for qs in cluster_quarter_jds.values() for q in qs.keys())))

    # Step 2: Classify skills per cluster
    print("\n[2] Classifying skills (foundation / trending / emerging)...")
    cluster_skill_forecasts = classify_skills(cluster_quarter_skills, cluster_quarter_jds)

    # Print sample
    sample_cluster = "US | Engineering - General"
    if sample_cluster in cluster_skill_forecasts:
        sf = cluster_skill_forecasts[sample_cluster]
        cats = defaultdict(int)
        for s, d in sf.items():
            cats[d['category']] += 1
        print("   Sample (%s): %s" % (sample_cluster, dict(cats)))
        # Show top foundation
        foundation = [(s, d) for s, d in sf.items() if d['category'] == 'foundation']
        foundation.sort(key=lambda x: -x[1]['predicted_strength'])
        print("   Foundation skills: %s" % [s for s, _ in foundation[:8]])
        trending = [(s, d) for s, d in sf.items() if d['category'] == 'trending_up']
        trending.sort(key=lambda x: -x[1]['predicted_strength'])
        print("   Trending up: %s" % [s for s, _ in trending[:5]])

    # Step 3: Generate predictions for Q3 & Q4 2026
    print("\n[3] Generating predictions for Q3 & Q4 2026...")
    target_quarters = ["2026Q3", "2026Q4"]
    predictions = generate_predictions(
        cur, cluster_skill_forecasts, cluster_quarter_skills, cluster_quarter_jds, target_quarters
    )
    print("   Generated %d predictions (%d clusters x %d quarters)" % (
        len(predictions), len(cluster_skill_forecasts), len(target_quarters)))

    # Step 4: Save to DB
    print("\n[4] Saving to predicted_cluster_skills table...")
    inserted = save_predictions(cur, conn, predictions)
    print("   Inserted/updated: %d rows" % inserted)

    # Print some examples
    print("\n" + "=" * 60)
    print("SAMPLE PREDICTIONS:")
    print("=" * 60)
    for pred in predictions[:5]:
        print("\n  %s (%s):" % (pred['cluster_name'], pred['quarter']))
        print("    Skills: %s" % pred['predicted_skills'][:6])
        print("    Predicted foundation: %s" % pred['skill_categories']['predicted']['foundation'][:4])
        print("    Historical core: %s" % pred['skill_categories']['historical']['core'][:4])
        print("    Confidence: %s" % pred['confidence'])

    cur.close()
    conn.close()
    print("\n\nDone!")


if __name__ == "__main__":
    main()
