"""
Build cluster-to-division mapping from job_descriptions table.
Maps each cluster to its division(s): Talent-on-Demand, Services, or Both.
Stores result in demand_forecasts-adjacent table and as JSON for the API.
"""

import pickle
import json
import mysql.connector
from collections import defaultdict
from pathlib import Path

DB_CONFIG = {
    "host": "localhost",
    "port": 3305,
    "user": "resume_user",
    "password": "resume_password",
    "database": "resume_processing",
}

DATA_DIR = Path(__file__).parent.parent / "data"


def load_title_to_cluster():
    with open(DATA_DIR / "title_to_cluster.pkl", "rb") as f:
        df = pickle.load(f)
    # DataFrame with columns: raw_title, role_cluster
    return dict(zip(df["raw_title"].str.lower(), df["role_cluster"]))


def get_jobs_with_division(conn):
    """Get all jobs with their title and division."""
    cursor = conn.cursor(dictionary=True)
    cursor.execute("""
        SELECT title, division_name, country 
        FROM job_descriptions 
        WHERE division_name IN ('Talent-on-Demand', 'Services')
        AND title IS NOT NULL
    """)
    rows = cursor.fetchall()
    cursor.close()
    return rows


def normalize_title(title):
    """Normalize title for matching — strip region prefixes, seniority suffixes."""
    import re
    t = title.strip().lower()
    # Remove region prefixes like "US - ", "IT - ", "IN - "
    t = re.sub(r'^[a-z]{2}\s*-\s*', '', t)
    # Remove seniority suffixes like " I", " II", " III", " IV"
    t = re.sub(r'\s+(i{1,3}|iv|v|vi|1|2|3|4)$', '', t)
    return t


def build_mapping(jobs, title_to_cluster):
    """
    For each cluster, count how many jobs come from ToD vs Services.
    Returns: { cluster_name: { 'Talent-on-Demand': count, 'Services': count } }
    """
    # Build lowercase lookup (also normalized)
    t2c_lower = {}
    for k, v in title_to_cluster.items():
        t2c_lower[k] = v  # already lowercased from load
        t2c_lower[normalize_title(k)] = v  # also store normalized version
    
    cluster_division_counts = defaultdict(lambda: defaultdict(int))
    matched = 0
    unmatched = 0

    for job in jobs:
        title_raw = job["title"].strip().lower()
        title_norm = normalize_title(job["title"])
        division = job["division_name"]
        
        cluster = t2c_lower.get(title_raw) or t2c_lower.get(title_norm)
        if cluster:
            cluster_division_counts[cluster][division] += 1
            matched += 1
        else:
            unmatched += 1

    print(f"Matched: {matched}, Unmatched: {unmatched}")
    return dict(cluster_division_counts)


def assign_division(counts, threshold=0.8):
    """
    Assign primary division to each cluster.
    If >80% of jobs are one division → assign that one.
    Otherwise → 'Both'
    """
    result = {}
    for cluster, divs in counts.items():
        tod = divs.get("Talent-on-Demand", 0)
        svc = divs.get("Services", 0)
        total = tod + svc
        
        if total == 0:
            result[cluster] = {"division": "Unknown", "tod_pct": 0, "svc_pct": 0, "total_jobs": 0}
            continue
            
        tod_pct = tod / total
        svc_pct = svc / total
        
        if tod_pct >= threshold:
            division = "Talent-on-Demand"
        elif svc_pct >= threshold:
            division = "Services"
        else:
            division = "Both"
        
        result[cluster] = {
            "division": division,
            "tod_count": tod,
            "svc_count": svc,
            "tod_pct": round(tod_pct * 100, 1),
            "svc_pct": round(svc_pct * 100, 1),
            "total_jobs": total,
        }
    
    return result


def save_to_db(conn, cluster_divisions):
    """Save cluster division mapping to a table."""
    cursor = conn.cursor()
    
    # Create table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS cluster_division_map (
            cluster_name VARCHAR(255) PRIMARY KEY,
            division VARCHAR(50) NOT NULL,
            tod_count INT DEFAULT 0,
            svc_count INT DEFAULT 0,
            tod_pct FLOAT DEFAULT 0,
            svc_pct FLOAT DEFAULT 0,
            total_jobs INT DEFAULT 0
        )
    """)
    
    # Clear and insert
    cursor.execute("TRUNCATE TABLE cluster_division_map")
    
    for cluster, info in cluster_divisions.items():
        cursor.execute("""
            INSERT INTO cluster_division_map 
            (cluster_name, division, tod_count, svc_count, tod_pct, svc_pct, total_jobs)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (
            cluster, info["division"],
            info.get("tod_count", 0), info.get("svc_count", 0),
            info.get("tod_pct", 0), info.get("svc_pct", 0),
            info.get("total_jobs", 0)
        ))
    
    conn.commit()
    cursor.close()
    print(f"✅ Saved {len(cluster_divisions)} clusters to cluster_division_map table")


def main():
    print("=" * 60)
    print("BUILDING CLUSTER → DIVISION MAPPING")
    print("=" * 60)
    
    # Load data
    title_to_cluster = load_title_to_cluster()
    print(f"Loaded title_to_cluster: {len(title_to_cluster)} titles")
    
    conn = mysql.connector.connect(**DB_CONFIG)
    jobs = get_jobs_with_division(conn)
    print(f"Jobs with division: {len(jobs)}")
    
    # Build mapping
    counts = build_mapping(jobs, title_to_cluster)
    print(f"Clusters found: {len(counts)}")
    
    # Assign divisions
    cluster_divisions = assign_division(counts)
    
    # Summary
    tod_only = sum(1 for v in cluster_divisions.values() if v["division"] == "Talent-on-Demand")
    svc_only = sum(1 for v in cluster_divisions.values() if v["division"] == "Services")
    both = sum(1 for v in cluster_divisions.values() if v["division"] == "Both")
    
    print(f"\n📊 Division Assignment (threshold=80%):")
    print(f"  Talent-on-Demand only: {tod_only}")
    print(f"  Services only:         {svc_only}")
    print(f"  Both:                  {both}")
    print(f"  Total clusters:        {len(cluster_divisions)}")
    
    # Print details
    print(f"\n{'Cluster':<35} {'Division':<20} {'ToD%':<8} {'Svc%':<8} {'Jobs'}")
    print("-" * 85)
    for cluster, info in sorted(cluster_divisions.items(), key=lambda x: x[1]["total_jobs"], reverse=True):
        print(f"{cluster:<35} {info['division']:<20} {info['tod_pct']:<8} {info['svc_pct']:<8} {info['total_jobs']}")
    
    # Save to DB
    save_to_db(conn, cluster_divisions)
    
    # Also save as JSON for reference
    out_path = DATA_DIR / "cluster_division_map.json"
    with open(out_path, "w") as f:
        json.dump(cluster_divisions, f, indent=2)
    print(f"📄 JSON saved: {out_path}")
    
    conn.close()


if __name__ == "__main__":
    main()
