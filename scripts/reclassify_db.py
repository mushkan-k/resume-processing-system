"""
Reclassify all updated_job_records with the new title_to_cluster_v2.pkl mapping.
Then re-run the actuals loading into demand_forecasts.
"""
import mysql.connector
import pandas as pd
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
DB_CONFIG = {
    "host": "localhost", "port": 3305, "database": "resume_processing",
    "user": "resume_user", "password": "resume_password",
}

print("=" * 70)
print("  RECLASSIFY ALL RECORDS WITH NEW CLUSTERS")
print("=" * 70)

# Load new mapping
mapping = pd.read_pickle(os.path.join(DATA_DIR, 'title_to_cluster.pkl'))
title_to_role = dict(zip(mapping['raw_title'], mapping['role_cluster']))
print(f"\n  New mapping: {len(title_to_role)} titles -> {len(set(title_to_role.values()))} clusters")

# Connect to DB
conn = mysql.connector.connect(**DB_CONFIG)
cur = conn.cursor(dictionary=True)

# Check if role_cluster column exists
cur.execute("SHOW COLUMNS FROM updated_job_records LIKE 'role_cluster'")
if not cur.fetchone():
    cur.execute("ALTER TABLE updated_job_records ADD COLUMN role_cluster VARCHAR(100)")
    conn.commit()
    print("  Added role_cluster column")

# Get all distinct titles
cur.execute("SELECT DISTINCT title FROM updated_job_records")
db_titles = [r['title'] for r in cur.fetchall()]
print(f"  Distinct titles in DB: {len(db_titles)}")

# Build update mapping
mapped = 0
unmapped = 0
updates = {}

for title in db_titles:
    cluster = title_to_role.get(title)
    if not cluster:
        # Try partial matching
        title_lower = title.lower() if title else ''
        for raw_t, c in title_to_role.items():
            if raw_t.lower() in title_lower or title_lower in raw_t.lower():
                cluster = c
                break
    if cluster:
        updates[title] = cluster
        mapped += 1
    else:
        unmapped += 1

print(f"  Mapped: {mapped}, Unmapped: {unmapped}")

# Batch update
print(f"\n  Updating role_cluster for {len(updates)} titles...")
batch_size = 500
titles_list = list(updates.items())
updated_total = 0

for i in range(0, len(titles_list), batch_size):
    batch = titles_list[i:i+batch_size]
    for title, cluster in batch:
        cur.execute("UPDATE updated_job_records SET role_cluster = %s WHERE title = %s", (cluster, title))
    conn.commit()
    updated_total += len(batch)
    if (i // batch_size) % 10 == 0:
        print(f"    ... {updated_total}/{len(titles_list)} titles updated")

print(f"  Updated {updated_total} title mappings")

# Verify
cur.execute("""
    SELECT role_cluster, COUNT(*) as cnt, SUM(openings) as opens
    FROM updated_job_records 
    WHERE issue_date >= '2026-01-01' AND role_cluster IS NOT NULL
    GROUP BY role_cluster ORDER BY opens DESC
""")
results = cur.fetchall()
print(f"\n  2026 CLUSTER DISTRIBUTION (after reclassification):")
print(f"  {'Cluster':<45} {'Records':>8} {'Openings':>9}")
print(f"  {'-'*45} {'-'*8} {'-'*9}")
for r in results:
    print(f"  {r['role_cluster']:<45} {r['cnt']:>8} {int(r['opens']):>9}")

cur.execute("SELECT COUNT(*) as c FROM updated_job_records WHERE issue_date >= '2026-01-01' AND role_cluster IS NULL")
null_count = cur.fetchone()['c']
print(f"\n  Records with no cluster (2026): {null_count}")

cur.close()
conn.close()
print(f"\n  Reclassification complete!")
