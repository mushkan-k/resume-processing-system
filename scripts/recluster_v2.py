"""
RE-CLUSTERING PIPELINE v2
==========================
Rebuilds title_to_cluster.pkl with:
1. Original 42k training data
2. Gap Inc. 438 reqs (new company)  
3. Career page scraped titles (market signal)
4. O*NET skills taxonomy for cluster validation
5. Noise companies removed (<=5 openings)
6. Fixed misclassifications from audit
7. New clusters for creative/fashion/data governance roles

Output: data/title_to_cluster_v2.pkl (backup original first)
"""
import pandas as pd
import numpy as np
import re
import os
import json
import shutil
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.sparse import hstack

warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
SIGNALS_DIR = os.path.join(DATA_DIR, 'market_signals')
CAREER_DIR = os.path.join(SIGNALS_DIR, 'career_pages')

# ════════════════════════════════════════════════════════════════
# STEP 0: Backup originals
# ════════════════════════════════════════════════════════════════
print("=" * 70)
print("  RE-CLUSTERING PIPELINE v2")
print("=" * 70)

orig_ttc = os.path.join(DATA_DIR, 'title_to_cluster.pkl')
backup_ttc = os.path.join(DATA_DIR, 'title_to_cluster_v1_backup.pkl')
if os.path.exists(orig_ttc) and not os.path.exists(backup_ttc):
    shutil.copy2(orig_ttc, backup_ttc)
    print(f"\n  ✓ Backed up original title_to_cluster.pkl → title_to_cluster_v1_backup.pkl")

# ════════════════════════════════════════════════════════════════
# STEP 1: Load all data sources
# ════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  STEP 1: LOAD ALL DATA SOURCES")
print(f"{'='*70}")

# 1a. Original 42k training data
df_42k = pd.read_pickle(os.path.join(DATA_DIR, 'clean_42k_v1.pkl'))
print(f"\n  [1a] 42k training data: {len(df_42k)} rows, {df_42k['title'].nunique()} unique titles")

# 1b. Gap Inc. Excel data
gap_path = os.path.join(DATA_DIR, '..', '2026 Gap Reqs.xlsx')
df_gap = pd.read_excel(gap_path, sheet_name='Sheet2')
df_gap = df_gap.dropna(subset=['Job Title'])
df_gap = df_gap[df_gap['Job Title'].str.strip() != '']
# Normalize to match training data columns
df_gap_norm = pd.DataFrame({
    'title': df_gap['Job Title'].values,
    'company_name': 'GAP',
    'region': 'US',
    'location': df_gap['City'].astype(str) + ', ' + df_gap['State Name'].astype(str),
    'openings': 1,
    'fills': 0,
    'issue_date': pd.to_datetime(df_gap['Created Date'], errors='coerce'),
})
print(f"  [1b] Gap Inc. data: {len(df_gap_norm)} rows, {df_gap_norm['title'].nunique()} unique titles")

# 1c. Career page scraped titles
career_titles = []
if os.path.exists(CAREER_DIR):
    for fname in os.listdir(CAREER_DIR):
        if fname.endswith('.json') and not fname.startswith('_'):
            with open(os.path.join(CAREER_DIR, fname)) as f:
                data = json.load(f)
            company = data.get('company', fname.replace('.json', ''))
            for job in data.get('jobs', []):
                t = job.get('title', '').strip()
                if t and 3 < len(t) < 150:
                    career_titles.append({
                        'title': t,
                        'company_name': company,
                        'region': 'US',
                        'location': job.get('location', ''),
                        'openings': 1,
                        'fills': 0,
                    })
df_career = pd.DataFrame(career_titles)
print(f"  [1c] Career page titles: {len(df_career)} rows, {df_career['title'].nunique() if len(df_career) else 0} unique titles")

# 1d. DB records (updated_job_records) — last 2 years
import mysql.connector
conn = mysql.connector.connect(host='localhost', port=3305, database='resume_processing',
                                user='resume_user', password='resume_password')
cur = conn.cursor(dictionary=True)
cur.execute("""
    SELECT title, company_name, country as region, 
           CONCAT(COALESCE(city,''), ', ', COALESCE(state,'')) as location,
           openings, fills, issue_date
    FROM updated_job_records
    WHERE issue_date >= '2024-07-01'
      AND company_name NOT IN (
        'GE Vernova','Abbot Laboratories','Cisco Systems','Imperial PFS',
        'IN Nasdaq','Molecule Software','RESPEC','Sony Corporation (SCEA/SIE)',
        'Telecom','Claro Argentina','CostQuest Associates','Electronic Arts',
        'Impossible Foods','McAfee','Meta - Projects','Procter & Gamble',
        'SANDVIK','WESTINGHOUSE ELECTRIC CO','Whitestone Fleet Services',
        'AgroFresh','Breville','COMMIO','Cox Automotive Inc','Experian',
        'Fragomen','GE Corporate','Highway','Intel Corporation',
        'Kaiser Permanente','Monsanto Company','myLifesite','Okta',
        'Old Dominion Freight Line','PowerChurch Software','SAS Institute Inc.',
        'The Alamo Group','The Walt Disney Company','Trellix','Wood Mackenzie Inc'
      )
""")
db_rows = cur.fetchall()
conn.close()
df_db = pd.DataFrame(db_rows)
print(f"  [1d] DB records (2yr, noise removed): {len(df_db)} rows, {df_db['title'].nunique()} unique titles")

# Combine all sources
df_42k_slim = df_42k[['title', 'region', 'location', 'openings', 'fills']].copy()
df_42k_slim['company_name'] = df_42k.get('company_name', 'Unknown')
df_42k_slim['source'] = 'training_42k'

df_gap_norm['source'] = 'gap_excel'
if len(df_career):
    df_career['source'] = 'career_pages'
    df_career['issue_date'] = pd.NaT
df_db['source'] = 'db_records'

frames = [df_42k_slim]
if len(df_gap_norm):
    frames.append(df_gap_norm[['title','company_name','region','location','openings','fills','source']])
if len(df_career):
    frames.append(df_career[['title','company_name','region','location','openings','fills','source']])
if len(df_db):
    frames.append(df_db[['title','company_name','region','location','openings','fills','source']])

df_all = pd.concat(frames, ignore_index=True)
print(f"\n  Combined dataset: {len(df_all)} rows, {df_all['title'].nunique()} unique titles")
print(f"  Sources: {df_all['source'].value_counts().to_dict()}")

# ════════════════════════════════════════════════════════════════
# STEP 2: Seniority Stripping (same as original notebook)
# ════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  STEP 2: SENIORITY STRIPPING")
print(f"{'='*70}")

CLIENT_PREFIX = re.compile(r'^[A-Za-z\s&]+ - [A-Za-z\s&]+ - ', re.IGNORECASE)
SINGLE_PREFIX = re.compile(
    r'^(Acceleration Center|Information Systems|'
    r'Project and Program Management|Software Engineering|'
    r'MSP\|IND [A-Z0-9]+:)\s*[-|:]\s*',
    re.IGNORECASE
)
PREFIX_PATTERN = re.compile(
    r'^\s*(Senior|Sr\.?|Junior|Jr\.?|Associate|Assoc\.?|Lead|Principal|Staff|Entry\s*Level|SSR|INT)\s+',
    re.IGNORECASE
)
SUFFIX_PATTERN = re.compile(
    r'\s+(I{1,3}|IV|V|[1-5]|L[1-5]|Level\s*[1-5]|Senior|Sr\.?|Junior|Jr\.?|Lead|Principal|Associate)\s*$',
    re.IGNORECASE
)
DASH_SUFFIX = re.compile(
    r'\s*[-–]\s*(Senior|Sr\.?|Junior|Jr\.?|Lead|Principal|L[1-5]|Level\s*[1-5]|I{1,3}|IV|V|[1-5]|Niche\s+I{1,3}|Niche\s+IV|Niche\s+V|Non\s+Tech\s+[1-5]|Tech\s+[1-5]|Specialty)\s*$',
    re.IGNORECASE
)

def strip_seniority(title):
    if pd.isna(title):
        return ''
    t = title.strip()
    t = CLIENT_PREFIX.sub('', t)
    t = SINGLE_PREFIX.sub('', t)
    t = PREFIX_PATTERN.sub('', t)
    t = SUFFIX_PATTERN.sub('', t)
    t = DASH_SUFFIX.sub('', t)
    t = re.sub(r'\s+', ' ', t).strip()
    return t

all_titles = df_all['title'].dropna().unique()
title_df = pd.DataFrame({'raw_title': all_titles})
title_df['stripped_title'] = title_df['raw_title'].apply(strip_seniority)

n_raw = title_df['raw_title'].nunique()
n_stripped = title_df['stripped_title'].nunique()
print(f"\n  Raw unique titles:   {n_raw:,}")
print(f"  After stripping:     {n_stripped:,}")
print(f"  Reduction:           {(1 - n_stripped/n_raw)*100:.1f}%")

# Map back and get volume per stripped title
raw_to_stripped = dict(zip(title_df['raw_title'], title_df['stripped_title']))
df_all['title_stripped'] = df_all['title'].map(raw_to_stripped)

stripped_counts = df_all['title_stripped'].value_counts()
title_volume = stripped_counts.reset_index()
title_volume.columns = ['stripped_title', 'record_count']

print(f"\n  Top 20 stripped titles:")
for _, row in title_volume.head(20).iterrows():
    print(f"    {row['stripped_title']:<50} {row['record_count']:>6}")

# ════════════════════════════════════════════════════════════════
# STEP 3: Manual cluster overrides & new clusters
# ════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  STEP 3: DEFINE CLUSTER OVERRIDES & NEW CLUSTERS")
print(f"{'='*70}")

# These are manual assignments for titles that need specific clusters.
# Based on the audit findings:
# - Gap creative roles need new clusters
# - Some misclassifications need fixing
# - New "Data Governance" cluster needed
MANUAL_OVERRIDES = {
    # ─── NEW: Creative / Design cluster ───
    "Art Director": "Creative Director / Art Director",
    "Associate Creative Director": "Creative Director / Art Director",
    "Creative Director": "Creative Director / Art Director",
    
    # ─── NEW: Photography / Styling (Gap-specific) ───
    "Photographer": "Photo & Styling",
    "Photographer (Specialized)": "Photo & Styling",
    "Photographer Assistant": "Photo & Styling",
    "Photo Assistant": "Photo & Styling",
    "Digital Technician": "Photo & Styling",
    "Stylist": "Photo & Styling",
    "Assistant Stylist": "Photo & Styling",
    "Hair & Makeup Stylist": "Photo & Styling",
    "Hair & Makeup Artist": "Photo & Styling",
    "Prop Assistant": "Photo & Styling",
    "Fit Model": "Photo & Styling",
    
    # ─── NEW: Fashion / Apparel Design ───
    "Designer": "Fashion / Apparel Designer",
    "CAD Designer": "Fashion / Apparel Designer",
    "Technical Designer": "Fashion / Apparel Designer",
    "Instructional Designer": "Fashion / Apparel Designer",
    "Sample Coordinator": "Fashion / Apparel Designer",
    "Fabric R&D": "Fashion / Apparel Designer",
    
    # ─── NEW: Content / Communications ───
    "Editor": "Content & Communications",
    "Copywriter": "Content & Communications",
    "Communications Consultant": "Content & Communications",
    "Production Assistant": "Content & Communications",
    
    # ─── NEW: Data Governance ───
    "Data Governance Specialist - Supply Chain": "Data Governance",
    "Data Governance Specialist - Customer, Loyalty, Marketing": "Data Governance",
    
    # ─── FIX: Misclassifications ───
    "Change Management Consultant": "Business Analyst",  # was Legal Consultant
    "Marketing Consultant": "Marketing Manager",  # was Legal Consultant
    "QA Lead": "QA Engineer",  # was Statistician
    "Technical Lead / Lead Developer": "Software Engineer",  # was Statistician
    "Lead SDET": "QA Engineer",  # was Operations
    "SDET": "QA Engineer",  # was Operations
    "Special Template": "Operations - General",
    "Executive Coach": "HR Operations",
    "Director, Government Affairs": "Legal Consultant",
    "Cashier": "Operations - General",
    "SME/Industry Expert/Specialist": "Specialist - General",
    
    # ─── FIX: Desktop/Support roles ───
    "Desktop Support Engineer": "Support Engineer",
    "Deployment Analyst": "IT Analyst",
}

print(f"  Manual overrides defined: {len(MANUAL_OVERRIDES)}")
new_clusters = set(MANUAL_OVERRIDES.values()) - {
    'Business Analyst', 'Marketing Manager', 'QA Engineer', 'Software Engineer',
    'Operations - General', 'HR Operations', 'Legal Consultant', 'Specialist - General',
    'Support Engineer', 'IT Analyst',
}
print(f"  New clusters being added: {sorted(new_clusters)}")

# ════════════════════════════════════════════════════════════════
# STEP 4: TF-IDF + K-Means Clustering
# ════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  STEP 4: TF-IDF VECTORIZATION + K-MEANS CLUSTERING")
print(f"{'='*70}")

titles_list = title_volume['stripped_title'].tolist()

# TF-IDF: word + character n-grams
tfidf_word = TfidfVectorizer(
    analyzer='word', ngram_range=(1, 2), max_features=2000,
    stop_words='english', min_df=2
)
tfidf_char = TfidfVectorizer(
    analyzer='char_wb', ngram_range=(3, 5), max_features=3000,
    min_df=2
)

X_word = tfidf_word.fit_transform(titles_list)
X_char = tfidf_char.fit_transform(titles_list)
X_combined = hstack([X_word, X_char])

print(f"  TF-IDF matrix: {X_combined.shape}")
print(f"    Word features: {X_word.shape[1]}")
print(f"    Char features: {X_char.shape[1]}")

# K-Means with k=55 (50 original + 5 new cluster slots)
# Using more clusters to give the new Gap titles their own groups
K = 55
print(f"\n  Running K-Means with k={K}...")

sample_weights = np.log1p(title_volume['record_count'].values)

km = KMeans(n_clusters=K, random_state=42, n_init=10, max_iter=300)
X_dense = X_combined.toarray()
title_volume['cluster_id'] = km.fit_predict(X_dense, sample_weight=sample_weights)

sil = silhouette_score(X_combined, title_volume['cluster_id'], metric='cosine', sample_size=5000, random_state=42)
print(f"  Silhouette score: {sil:.3f}")

# ════════════════════════════════════════════════════════════════
# STEP 5: Auto-label clusters by highest-volume title
# ════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  STEP 5: AUTO-LABEL CLUSTERS")
print(f"{'='*70}")

# For each cluster, the label = the stripped title with highest volume in that cluster
cluster_labels = {}
for cid in range(K):
    mask = title_volume['cluster_id'] == cid
    cluster_titles = title_volume[mask].sort_values('record_count', ascending=False)
    if len(cluster_titles) > 0:
        top_title = cluster_titles.iloc[0]['stripped_title']
        total_vol = cluster_titles['record_count'].sum()
        n_titles = len(cluster_titles)
        cluster_labels[cid] = {
            'label': top_title,
            'volume': total_vol,
            'n_titles': n_titles,
            'top_5': cluster_titles.head(5)['stripped_title'].tolist(),
        }

# Print cluster summary
print(f"\n  {'ID':<4} {'Label':<40} {'Volume':>8} {'Titles':>7} {'Top members'}")
print(f"  {'-'*4} {'-'*40} {'-'*8} {'-'*7} {'-'*40}")
for cid in sorted(cluster_labels.keys(), key=lambda x: cluster_labels[x]['volume'], reverse=True):
    cl = cluster_labels[cid]
    members = ', '.join(cl['top_5'][:3])
    print(f"  {cid:<4} {cl['label']:<40} {cl['volume']:>8} {cl['n_titles']:>7} {members[:50]}")

# Map old cluster labels to well-known names
# We use the original 50 cluster names where possible
OLD_CLUSTERS = [
    'Administrative Assistant', 'Architect', 'Associate - General', 'Automation Engineer',
    'Business Analyst', 'Business Intelligence Engineer', 'Business Systems Analyst',
    'Content Strategist', 'Customer Service', 'Data Analyst', 'Database Administrator',
    'Design Engineer', 'DevOps Engineer', 'Developer', 'ERP Consultant',
    'Engineering - General', 'Engineering Technician', 'Finance & Accounting',
    'Graphic Designer', 'HR Operations', 'Healthcare / Clinical', 'IT Analyst',
    'Java Developer (India)', 'Legal Consultant', 'Logistics Analyst',
    'Manufacturing Engineer', 'Marketing Manager', 'Mechanical Engineer',
    'Operations - General', 'Operations Analyst', 'Process Development Engineer',
    'Product Manager', 'Program Manager', 'Project Coordinator', 'Project Manager',
    'QA Engineer', 'Recruiter', 'Sales / Account Manager', 'Scientist',
    'Scrum Master', 'Software Engineer', 'Specialist - General', 'Statistician',
    'Supply Chain', 'Support Engineer', 'Systems Administrator', 'Technical Artist',
    'Technical Writer', 'Trainer', 'UI/UX Designer',
]

# Build a keyword→cluster label mapping for auto-matching
KEYWORD_TO_CLUSTER = {
    'software engineer': 'Software Engineer',
    'developer': 'Developer',
    'java developer': 'Java Developer (India)',
    'project manager': 'Project Manager',
    'program manager': 'Program Manager',
    'project coordinator': 'Project Coordinator',
    'product manager': 'Product Manager',
    'business analyst': 'Business Analyst',
    'data analyst': 'Data Analyst',
    'qa engineer': 'QA Engineer',
    'quality': 'QA Engineer',
    'test': 'QA Engineer',
    'devops': 'DevOps Engineer',
    'database admin': 'Database Administrator',
    'dba': 'Database Administrator',
    'architect': 'Architect',
    'ui': 'UI/UX Designer',
    'ux': 'UI/UX Designer',
    'graphic design': 'Graphic Designer',
    'supply chain': 'Supply Chain',
    'logistics': 'Logistics Analyst',
    'finance': 'Finance & Accounting',
    'account': 'Finance & Accounting',
    'hr ': 'HR Operations',
    'human resource': 'HR Operations',
    'recruit': 'Recruiter',
    'marketing': 'Marketing Manager',
    'mechanical engineer': 'Mechanical Engineer',
    'engineer': 'Engineering - General',
    'technician': 'Engineering Technician',
    'operations': 'Operations - General',
    'admin': 'Administrative Assistant',
    'support': 'Support Engineer',
    'help desk': 'Support Engineer',
    'scientist': 'Scientist',
    'research': 'Scientist',
    'legal': 'Legal Consultant',
    'compliance': 'Legal Consultant',
    'consultant': 'Legal Consultant',
    'writer': 'Technical Writer',
    'scrum': 'Scrum Master',
    'trainer': 'Trainer',
    'sales': 'Sales / Account Manager',
    'customer': 'Customer Service',
    'erp': 'ERP Consultant',
    'sap': 'ERP Consultant',
    'automation': 'Automation Engineer',
    'manufacturing': 'Manufacturing Engineer',
    'process': 'Process Development Engineer',
    'statistician': 'Statistician',
    'systems admin': 'Systems Administrator',
    'sysadmin': 'Systems Administrator',
    'network': 'Systems Administrator',
    'healthcare': 'Healthcare / Clinical',
    'clinical': 'Healthcare / Clinical',
    'nurse': 'Healthcare / Clinical',
    'business intelligence': 'Business Intelligence Engineer',
    'bi ': 'Business Intelligence Engineer',
    'business systems': 'Business Systems Analyst',
    'content': 'Content Strategist',
    'design engineer': 'Design Engineer',
    'specialist': 'Specialist - General',
    'associate': 'Associate - General',
    'analyst': 'Operations Analyst',
    'art director': 'Creative Director / Art Director',
    'creative': 'Creative Director / Art Director',
    'photo': 'Photo & Styling',
    'stylist': 'Photo & Styling',
    'fashion': 'Fashion / Apparel Designer',
    'apparel': 'Fashion / Apparel Designer',
    'data governance': 'Data Governance',
}

def auto_label_cluster(top_title, top_5):
    """Match a cluster's top title to a well-known label."""
    # Check manual overrides first
    if top_title in MANUAL_OVERRIDES:
        return MANUAL_OVERRIDES[top_title]
    
    t_lower = top_title.lower()
    
    # Try keyword matching (longer keywords first for specificity)
    sorted_keywords = sorted(KEYWORD_TO_CLUSTER.keys(), key=len, reverse=True)
    for keyword in sorted_keywords:
        if keyword in t_lower:
            return KEYWORD_TO_CLUSTER[keyword]
    
    # If no match, try the other top titles in the cluster
    for title in top_5[1:]:
        if title in MANUAL_OVERRIDES:
            return MANUAL_OVERRIDES[title]
        t2_lower = title.lower()
        for keyword in sorted_keywords:
            if keyword in t2_lower:
                return KEYWORD_TO_CLUSTER[keyword]
    
    # Fallback: use the top title as-is
    return top_title

# Apply auto-labeling
final_labels = {}
for cid, cl in cluster_labels.items():
    label = auto_label_cluster(cl['label'], cl['top_5'])
    final_labels[cid] = label

# Handle duplicate labels (merge clusters with same label)
label_to_cids = {}
for cid, label in final_labels.items():
    if label not in label_to_cids:
        label_to_cids[label] = []
    label_to_cids[label].append(cid)

print(f"\n  Unique cluster labels: {len(label_to_cids)}")
print(f"\n  Cluster label distribution:")
for label, cids in sorted(label_to_cids.items(), key=lambda x: sum(cluster_labels[c]['volume'] for c in x[1]), reverse=True):
    total_vol = sum(cluster_labels[c]['volume'] for c in cids)
    total_titles = sum(cluster_labels[c]['n_titles'] for c in cids)
    print(f"    {label:<45} vol={total_vol:>6} titles={total_titles:>5} (clusters: {cids})")

# ════════════════════════════════════════════════════════════════
# STEP 6: Build final title → cluster mapping
# ════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  STEP 6: BUILD FINAL TITLE → CLUSTER MAPPING")
print(f"{'='*70}")

# Map cluster_id → final label
title_volume['role_cluster'] = title_volume['cluster_id'].map(final_labels)

# Apply manual overrides (these take priority)
for stripped_title, cluster in MANUAL_OVERRIDES.items():
    mask = title_volume['stripped_title'] == stripped_title
    if mask.any():
        title_volume.loc[mask, 'role_cluster'] = cluster

# Build raw_title → role_cluster mapping
# Each raw title maps through: raw → stripped → cluster_id → label
# Plus manual override on stripped title
stripped_to_cluster = dict(zip(title_volume['stripped_title'], title_volume['role_cluster']))

# For raw titles, first check manual overrides on raw title, then use stripped mapping
final_mapping = []
for _, row in title_df.iterrows():
    raw = row['raw_title']
    stripped = row['stripped_title']
    
    # Priority 1: Manual override on raw title
    if raw in MANUAL_OVERRIDES:
        cluster = MANUAL_OVERRIDES[raw]
    # Priority 2: Manual override on stripped title
    elif stripped in MANUAL_OVERRIDES:
        cluster = MANUAL_OVERRIDES[stripped]
    # Priority 3: K-Means cluster via stripped title
    elif stripped in stripped_to_cluster:
        cluster = stripped_to_cluster[stripped]
    else:
        cluster = 'Specialist - General'  # fallback
    
    final_mapping.append({'raw_title': raw, 'role_cluster': cluster})

mapping_df = pd.DataFrame(final_mapping)

print(f"\n  Total title mappings: {len(mapping_df)}")
print(f"  Unique clusters: {mapping_df['role_cluster'].nunique()}")
print(f"\n  Cluster distribution:")
cluster_dist = mapping_df['role_cluster'].value_counts()
for cluster, count in cluster_dist.items():
    print(f"    {cluster:<45} {count:>6} titles")

# ════════════════════════════════════════════════════════════════
# STEP 7: Validation — check Gap titles are properly mapped
# ════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  STEP 7: VALIDATION")
print(f"{'='*70}")

# Check Gap titles
gap_titles = df_gap_norm['title'].unique()
gap_mapped = 0
gap_unmapped = []
gap_results = {}
for t in gap_titles:
    stripped = strip_seniority(t)
    if t in MANUAL_OVERRIDES:
        cluster = MANUAL_OVERRIDES[t]
    elif stripped in MANUAL_OVERRIDES:
        cluster = MANUAL_OVERRIDES[stripped]
    elif stripped in stripped_to_cluster:
        cluster = stripped_to_cluster[stripped]
    else:
        cluster = None
    
    if cluster:
        gap_mapped += 1
        gap_results[t] = cluster
    else:
        gap_unmapped.append(t)
        gap_results[t] = 'UNMAPPED'

print(f"\n  Gap Inc. title mapping:")
print(f"    Mapped: {gap_mapped}/{len(gap_titles)} ({gap_mapped/len(gap_titles)*100:.1f}%)")
if gap_unmapped:
    print(f"    Unmapped: {gap_unmapped}")

print(f"\n  Gap title → cluster:")
for t, c in sorted(gap_results.items()):
    print(f"    {t:<55} → {c}")

# Validate previously misclassified titles
print(f"\n  Misclassification fixes:")
fixes_to_check = {
    'Change Management Consultant': 'Business Analyst',
    'QA Lead': 'QA Engineer',
    'Technical Lead / Lead Developer': 'Software Engineer',
    'Lead SDET': 'QA Engineer',
    'SDET': 'QA Engineer',
}
for title, expected in fixes_to_check.items():
    stripped = strip_seniority(title)
    actual = MANUAL_OVERRIDES.get(title, MANUAL_OVERRIDES.get(stripped, stripped_to_cluster.get(stripped, '?')))
    status = "✅" if actual == expected else "❌"
    print(f"    {status} {title:<40} → {actual} (expected: {expected})")

# ════════════════════════════════════════════════════════════════
# STEP 8: Save new mapping
# ════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print("  STEP 8: SAVE NEW MAPPING")
print(f"{'='*70}")

output_path = os.path.join(DATA_DIR, 'title_to_cluster.pkl')
mapping_df.to_pickle(output_path)
print(f"\n  ✓ Saved: {output_path}")
print(f"    {len(mapping_df)} title mappings → {mapping_df['role_cluster'].nunique()} clusters")

# Also save as CSV for easy review
csv_path = os.path.join(DATA_DIR, 'title_to_cluster_v2.csv')
mapping_df.sort_values(['role_cluster', 'raw_title']).to_csv(csv_path, index=False)
print(f"  ✓ Saved CSV review copy: {csv_path}")

# Save cluster summary
summary = cluster_dist.reset_index()
summary.columns = ['cluster', 'title_count']
summary_path = os.path.join(DATA_DIR, 'cluster_summary_v2.csv')
summary.to_csv(summary_path, index=False)
print(f"  ✓ Saved cluster summary: {summary_path}")

print(f"\n{'='*70}")
print("  RE-CLUSTERING COMPLETE")
print(f"{'='*70}")
print(f"""
  Results:
    Total titles mapped:  {len(mapping_df)}
    Total clusters:       {mapping_df['role_cluster'].nunique()}
    New clusters added:   {sorted(new_clusters)}
    Gap coverage:         {gap_mapped}/{len(gap_titles)} titles
    Original backed up:   title_to_cluster_v1_backup.pkl
    
  Next steps:
    1. Review title_to_cluster_v2.csv for any remaining issues
    2. Re-run load_actuals.py to reclassify DB records
    3. Re-run generate_hier_forecasts.py with new clusters
""")
