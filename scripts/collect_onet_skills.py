"""
O*NET Skills Taxonomy Collector
================================
Free public data — https://www.onetcenter.org/database.html
Maps occupations to detailed skills, knowledge, abilities.
This is the gold standard for "what skills does this role need?"

Uses the O*NET Web Services API (free, needs registration).
Alternative: Download bulk files from https://www.onetcenter.org/database.html#all-files
"""
import requests
import json
import os
import time
from datetime import datetime

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'market_signals')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# O*NET Web Services base URL
ONET_BASE = "https://services.onetcenter.org/ws"

# SOC → O*NET code mapping (O*NET uses SOC with .00 suffix)
# Same mapping as BLS but with .00 appended
CLUSTER_TO_ONET = {
    "Software Engineer":        ["15-1252.00"],
    "Developer":                ["15-1252.00", "15-1254.00"],
    "DevOps Engineer":          ["15-1244.00"],
    "QA Engineer":              ["15-1253.00"],
    "Data Analyst":             ["15-2051.00"],
    "Business Analyst":         ["13-1111.00"],
    "Database Administrator":   ["15-1242.00"],
    "Architect":                ["15-1299.00"],
    "UI/UX Designer":           ["15-1255.00"],
    "Graphic Designer":         ["27-1024.00"],
    "Project Manager":          ["11-9199.00"],
    "Program Manager":          ["11-9199.00"],
    "Product Manager":          ["11-2021.00"],
    "Engineering - General":    ["17-2199.00"],
    "Mechanical Engineer":      ["17-2141.00"],
    "Operations - General":     ["11-1021.00"],
    "Supply Chain":             ["13-1081.00"],
    "Finance & Accounting":     ["13-2011.00"],
    "HR Operations":            ["13-1071.00"],
    "Support Engineer":         ["15-1232.00"],
    "Scientist":                ["19-1042.00"],
    "Creative Director":        ["27-1011.00"],
    "Photographer":             ["27-4021.00"],
    "Fashion Designer":         ["27-1022.00"],
    "Marketing Manager":        ["11-2021.00"],
    "Technical Writer":         ["27-3042.00"],
    "Statistician":             ["15-2041.00"],
    "Administrative Assistant": ["43-6014.00"],
    "Recruiter":                ["13-1071.00"],
}


def fetch_onet_skills(onet_code, auth=None):
    """
    Fetch skills for an O*NET occupation code.
    Without auth, uses the public browse (limited).
    With auth (username:password), uses the API.
    """
    headers = {"Accept": "application/json"}
    if auth:
        headers["Authorization"] = f"Basic {auth}"
    
    url = f"{ONET_BASE}/online/occupations/{onet_code}/summary/skills"
    
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        elif resp.status_code == 422 or resp.status_code == 401:
            # Try alternative: direct O*NET online scrape-safe endpoint
            return None
        else:
            return None
    except Exception as e:
        print(f"    Error fetching {onet_code}: {e}")
        return None


def fetch_onet_bulk_skills():
    """
    Download O*NET skills data from their bulk download files.
    These are CSV files publicly available.
    """
    # O*NET database bulk files (updated twice/year)
    # https://www.onetcenter.org/database.html#all-files
    bulk_urls = {
        "skills": "https://www.onetcenter.org/dl_files/database/db_29_1_text/Skills.txt",
        "knowledge": "https://www.onetcenter.org/dl_files/database/db_29_1_text/Knowledge.txt",
        "abilities": "https://www.onetcenter.org/dl_files/database/db_29_1_text/Abilities.txt",
        "tech_skills": "https://www.onetcenter.org/dl_files/database/db_29_1_text/Technology%20Skills.txt",
        "tools_used": "https://www.onetcenter.org/dl_files/database/db_29_1_text/Tools%20Used.txt",
    }
    
    results = {}
    
    for name, url in bulk_urls.items():
        print(f"  Downloading O*NET {name}...")
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                lines = resp.text.strip().split("\n")
                header = lines[0].split("\t")
                rows = [dict(zip(header, line.split("\t"))) for line in lines[1:]]
                
                # Filter to only our SOC codes
                our_codes = set()
                for codes in CLUSTER_TO_ONET.values():
                    our_codes.update(codes)
                
                filtered = [r for r in rows if r.get("O*NET-SOC Code", "") in our_codes]
                results[name] = filtered
                print(f"    ✓ {len(filtered)} entries (from {len(rows)} total)")
            elif resp.status_code == 403:
                print(f"    ⚠ Access restricted — need to accept license at onetcenter.org")
                print(f"      URL: {url}")
            else:
                print(f"    ✗ HTTP {resp.status_code}")
        except Exception as e:
            print(f"    ✗ Error: {e}")
        
        time.sleep(0.5)
    
    return results


def build_skills_taxonomy():
    """Build a cluster → skills mapping from O*NET data."""
    print("=" * 70)
    print("  O*NET SKILLS TAXONOMY COLLECTION")
    print("=" * 70)
    
    # Try bulk download first
    print("\n  [1] Attempting O*NET bulk data download...")
    bulk_data = fetch_onet_bulk_skills()
    
    if bulk_data:
        # Process into cluster → skills mapping
        taxonomy = {}
        
        for cluster, onet_codes in CLUSTER_TO_ONET.items():
            cluster_skills = {"skills": [], "knowledge": [], "tech_skills": [], "tools": []}
            
            for code in onet_codes:
                # Skills
                for entry in bulk_data.get("skills", []):
                    if entry.get("O*NET-SOC Code") == code:
                        skill_name = entry.get("Element Name", "")
                        importance = float(entry.get("Data Value", 0))
                        if importance >= 3.0:  # Only important skills
                            cluster_skills["skills"].append({
                                "name": skill_name,
                                "importance": importance,
                            })
                
                # Knowledge
                for entry in bulk_data.get("knowledge", []):
                    if entry.get("O*NET-SOC Code") == code:
                        k_name = entry.get("Element Name", "")
                        importance = float(entry.get("Data Value", 0))
                        if importance >= 3.0:
                            cluster_skills["knowledge"].append({
                                "name": k_name,
                                "importance": importance,
                            })
                
                # Tech skills
                for entry in bulk_data.get("tech_skills", []):
                    if entry.get("O*NET-SOC Code") == code:
                        cluster_skills["tech_skills"].append(entry.get("Example", ""))
                
                # Tools
                for entry in bulk_data.get("tools_used", []):
                    if entry.get("O*NET-SOC Code") == code:
                        cluster_skills["tools"].append(entry.get("Commodity Title", ""))
            
            # Deduplicate
            cluster_skills["skills"] = sorted(
                {s["name"]: s for s in cluster_skills["skills"]}.values(),
                key=lambda x: x["importance"], reverse=True
            )
            cluster_skills["knowledge"] = sorted(
                {k["name"]: k for k in cluster_skills["knowledge"]}.values(),
                key=lambda x: x["importance"], reverse=True
            )
            cluster_skills["tech_skills"] = sorted(set(cluster_skills["tech_skills"]))
            cluster_skills["tools"] = sorted(set(cluster_skills["tools"]))
            
            taxonomy[cluster] = cluster_skills
        
        # Save
        output = {
            "source": "O*NET OnLine (National Center for O*NET Development)",
            "collected_at": datetime.now().isoformat(),
            "cluster_skills": taxonomy,
        }
        
        output_path = os.path.join(OUTPUT_DIR, "onet_skills_taxonomy.json")
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  ✓ Saved to {output_path}")
        
        # Print summary
        print(f"\n  SKILLS PER CLUSTER:")
        for cluster, data in sorted(taxonomy.items()):
            s = len(data["skills"])
            k = len(data["knowledge"])
            t = len(data["tech_skills"])
            print(f"    {cluster:<35} skills={s} knowledge={k} tech={t}")
        
        return taxonomy
    
    print("\n  ⚠ Bulk download failed. Will try API approach...")
    return {}


if __name__ == "__main__":
    build_skills_taxonomy()
