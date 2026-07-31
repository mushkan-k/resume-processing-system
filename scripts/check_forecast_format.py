"""Check if forecasts (predicted) also use Region | Role combo format."""
import mysql.connector

conn = mysql.connector.connect(host='localhost', port=3305, database='resume_processing',
                                user='resume_user', password='resume_password')
cur = conn.cursor(dictionary=True)

cur.execute("""
    SELECT cluster_name, forecast_month, demand_predicted 
    FROM demand_forecasts 
    WHERE data_type='predicted' 
    ORDER BY demand_predicted DESC
    LIMIT 40
""")
rows = cur.fetchall()

print("PREDICTED cluster_name format (top 40 by demand):\n")
for r in rows:
    has_pipe = " | " in r['cluster_name']
    print(f"  {'✅' if has_pipe else '❌'} {r['cluster_name']:<50} {r['forecast_month']}  demand={int(r['demand_predicted'])}")

# Count format types
cur.execute("SELECT cluster_name FROM demand_forecasts WHERE data_type='predicted'")
all_pred = [r['cluster_name'] for r in cur.fetchall()]
with_region = [c for c in all_pred if ' | ' in c]
without_region = [c for c in all_pred if ' | ' not in c]

print(f"\n  With Region|Role format: {len(with_region)}")
print(f"  Without (plain role):    {len(without_region)}")

if without_region:
    print(f"\n  Plain role names in predictions:")
    for c in sorted(set(without_region)):
        print(f"    {c}")

conn.close()
