"""Quick check: which career page companies match training data."""
import pandas as pd, json, os, shutil

data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

# Restore backup
shutil.copy(os.path.join(data_dir, 'clean_42k_v1_pre_enrich_backup.pkl'), 
            os.path.join(data_dir, 'clean_42k_v1.pkl'))
print('Restored backup')

df = pd.read_pickle(os.path.join(data_dir, 'clean_42k_v1.pkl'))
companies = sorted(df['company_name'].unique())
print(f'{len(companies)} training companies')

cp_dir = os.path.join(data_dir, 'market_signals', 'career_pages')
for f in sorted(os.listdir(cp_dir)):
    if not f.endswith('.json'):
        continue
    data = json.load(open(os.path.join(cp_dir, f)))
    cp_name = data.get('company', f.replace('.json',''))
    jobs = len(data.get('jobs', []))
    
    # Smart matching
    cp_clean = cp_name.lower().replace(' ', '').replace(',', '').replace('.', '').replace('inc', '').replace('ltd', '')
    matches = []
    for c in companies:
        c_clean = c.lower().replace(' ', '').replace(',', '').replace('.', '').replace('inc', '').replace('ltd', '')
        if cp_clean in c_clean or c_clean in cp_clean:
            matches.append(c)
        elif cp_clean[:5] == c_clean[:5] and len(cp_clean) > 4:
            matches.append(c)
    
    status = matches[0] if matches else 'NO MATCH'
    print(f'  {cp_name:30s} ({jobs:3d} jobs) -> {status}')
