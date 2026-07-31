import mysql.connector
conn = mysql.connector.connect(host='localhost', port=3305, database='resume_processing', user='resume_user', password='resume_password')
cur = conn.cursor()

# Delete TFT predictions
cur.execute("DELETE FROM demand_forecasts WHERE model_used = 'TFT_v1_QuantileLoss' AND data_type = 'predicted'")
deleted = cur.rowcount
print(f'Deleted {deleted} TFT rows')
conn.commit()

# Check what's left
cur.execute("SELECT model_used, data_type, COUNT(*) as cnt FROM demand_forecasts GROUP BY model_used, data_type")
print()
for r in cur.fetchall():
    print(f'  {r[0]} | {r[1]} | {r[2]} rows')

# Check if Bayesian predictions still exist
cur.execute("SELECT COUNT(*) FROM demand_forecasts WHERE data_type = 'predicted'")
pred_count = cur.fetchone()[0]
print(f'\nPredicted rows remaining: {pred_count}')

if pred_count == 0:
    print('Need to re-run Bayesian forecasts!')
else:
    print('Bayesian predictions still in DB - good!')

conn.close()
