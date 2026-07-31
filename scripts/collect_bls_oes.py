"""
BLS OES (Occupational Employment & Wage Statistics) Data Collector
=================================================================
Free public API — no key needed for basic access, key for higher rate limits.
Gets employment counts, wages, and projections by occupation.

API docs: https://www.bls.gov/developers/
OES data: https://www.bls.gov/oes/
"""
import requests
import json
import os
import time
from datetime import datetime

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'market_signals')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── SOC codes mapped to our role clusters ───
# Standard Occupational Classification (SOC) codes
# https://www.bls.gov/soc/2018/major_groups.htm
CLUSTER_TO_SOC = {
    "Software Engineer":        ["15-1252"],  # Software Developers
    "Developer":                ["15-1252", "15-1254"],  # Software Devs + Web Devs
    "Java Developer (India)":   ["15-1252"],
    "DevOps Engineer":          ["15-1244"],  # Network & Computer Systems Admins (closest)
    "QA Engineer":              ["15-1253"],  # Software Quality Assurance Analysts
    "Data Analyst":             ["15-2051"],  # Data Scientists
    "Data Scientist":           ["15-2051"],
    "Business Analyst":         ["13-1111"],  # Management Analysts
    "Business Intelligence Engineer": ["15-2051"],
    "Database Administrator":   ["15-1242"],  # Database Administrators & Architects
    "Systems Administrator":    ["15-1244"],  # Network & Computer Systems Admins
    "Architect":                ["15-1299"],  # Computer Occupations, All Other
    "UI/UX Designer":           ["15-1255"],  # Web & Digital Interface Designers
    "Graphic Designer":         ["27-1024"],  # Graphic Designers
    "Project Manager":          ["11-9199"],  # Managers, All Other
    "Program Manager":          ["11-9199"],
    "Project Coordinator":      ["13-1082"],  # Project Management Specialists
    "Product Manager":          ["11-2021"],  # Marketing Managers
    "Engineering - General":    ["17-2199"],  # Engineers, All Other
    "Mechanical Engineer":      ["17-2141"],  # Mechanical Engineers
    "Engineering Technician":   ["17-3029"],  # Engineering Technicians
    "Operations - General":     ["11-1021"],  # General & Operations Managers
    "Supply Chain":             ["13-1081"],  # Logisticians
    "Logistics Analyst":        ["13-1081"],
    "Finance & Accounting":     ["13-2011"],  # Accountants & Auditors
    "HR Operations":            ["13-1071"],  # HR Specialists
    "Legal Consultant":         ["23-1011"],  # Lawyers (closest)
    "Recruiter":                ["13-1071"],  # HR Specialists
    "Administrative Assistant": ["43-6014"],  # Secretaries & Admin Assistants
    "Marketing Manager":        ["11-2021"],  # Marketing Managers
    "Technical Writer":         ["27-3042"],  # Technical Writers
    "Support Engineer":         ["15-1232"],  # Computer User Support Specialists
    "Scientist":                ["19-1042"],  # Medical Scientists (for pharma clients)
    "Statistician":             ["15-2041"],  # Statisticians
    "Automation Engineer":      ["17-2199"],  # Engineers, All Other
    "Process Development Engineer": ["17-2112"],  # Industrial Engineers
    "Associate - General":      ["43-9199"],  # Office & Admin Support, All Other
    "Specialist - General":     ["13-1199"],  # Business Operations Specialists, All Other
    # New clusters for Gap
    "Creative Director":        ["27-1011"],  # Art Directors
    "Photographer":             ["27-4021"],  # Photographers
    "Fashion Designer":         ["27-1022"],  # Fashion Designers
}

# BLS API series ID format for OES national data:
# OEUM003400000000001525200003  (complex)
# Simpler: use the public data API v2
BLS_API_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data/"

# OES series format: OEUN[area][industry][occupation][datatype]
# For national all-industries: area=0000000, industry=000000
# datatype: 01=employment, 04=mean hourly wage, 13=median hourly wage


def build_oes_series_id(soc_code, data_type="01"):
    """
    Build OES series ID.
    Format: OEUM + area(7) + industry(6) + occupation(6) + datatype(2)
    National, all industries: OEUN0000000000000{soc}00{datatype}
    """
    soc_clean = soc_code.replace("-", "")
    # National, cross-industry
    return f"OEUN000000000000{soc_clean}0{data_type}"


def fetch_bls_data(series_ids, start_year=2020, end_year=2025):
    """Fetch data from BLS API v2 (public, no key = 10 series max, 25 req/day)."""
    headers = {"Content-Type": "application/json"}
    payload = {
        "seriesid": series_ids[:10],  # Max 10 per request without API key
        "startyear": str(start_year),
        "endyear": str(end_year),
    }
    
    # If we have a BLS API key, use it (500 series, 500 req/day)
    bls_key = os.getenv("BLS_API_KEY")
    if bls_key:
        payload["registrationkey"] = bls_key
    
    resp = requests.post(BLS_API_URL, json=payload, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def collect_oes_data():
    """Collect OES employment and wage data for all mapped SOC codes."""
    print("=" * 70)
    print("  BLS OES DATA COLLECTION")
    print("=" * 70)
    
    # Get unique SOC codes
    all_socs = set()
    for socs in CLUSTER_TO_SOC.values():
        all_socs.update(socs)
    all_socs = sorted(all_socs)
    print(f"\n  Unique SOC codes to fetch: {len(all_socs)}")
    
    # Build series IDs: employment (01) and mean wage (04) for each SOC
    results = {}
    
    # Process in batches of 5 SOCs (= 10 series: employment + wage each)
    for batch_start in range(0, len(all_socs), 5):
        batch_socs = all_socs[batch_start:batch_start + 5]
        series_ids = []
        series_map = {}
        
        for soc in batch_socs:
            emp_id = build_oes_series_id(soc, "01")  # Employment
            wage_id = build_oes_series_id(soc, "04")  # Mean hourly wage
            series_ids.extend([emp_id, wage_id])
            series_map[emp_id] = (soc, "employment")
            series_map[wage_id] = (soc, "mean_hourly_wage")
        
        print(f"\n  Fetching batch: {batch_socs}...")
        try:
            data = fetch_bls_data(series_ids)
            
            if data.get("status") == "REQUEST_SUCCEEDED":
                for series in data.get("Results", {}).get("series", []):
                    sid = series["seriesID"]
                    if sid in series_map:
                        soc, dtype = series_map[sid]
                        if soc not in results:
                            results[soc] = {"soc_code": soc, "employment": {}, "mean_hourly_wage": {}}
                        for point in series.get("data", []):
                            year = point["year"]
                            period = point["period"]
                            if period == "M13":  # Annual average
                                val = point["value"]
                                results[soc][dtype][year] = float(val) if val != "-" else None
                                print(f"    {soc} | {dtype} | {year}: {val}")
            else:
                print(f"    API error: {data.get('message', 'Unknown')}")
                
        except Exception as e:
            print(f"    Error: {e}")
        
        time.sleep(1)  # Rate limiting
    
    # Save results
    output = {
        "source": "BLS OES (Occupational Employment & Wage Statistics)",
        "collected_at": datetime.now().isoformat(),
        "soc_to_cluster_mapping": {soc: [c for c, socs in CLUSTER_TO_SOC.items() if soc in socs] 
                                    for soc in all_socs},
        "data": results,
    }
    
    output_path = os.path.join(OUTPUT_DIR, "bls_oes_data.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  ✓ Saved to {output_path}")
    
    return results


def collect_bls_projections():
    """
    Collect BLS Employment Projections (10-year outlook).
    Series format: occupation projections from EP data.
    """
    print(f"\n{'='*70}")
    print("  BLS EMPLOYMENT PROJECTIONS (10-year outlook)")
    print(f"{'='*70}")
    
    # Employment projections use different series IDs
    # Format: EUUR[base/proj][soc code padded]00000000
    all_socs = set()
    for socs in CLUSTER_TO_SOC.values():
        all_socs.update(socs)
    
    projections = {}
    for soc in sorted(all_socs):
        soc_clean = soc.replace("-", "")
        # Try to get projection data
        series_ids = [
            f"EUUR00{soc_clean}00000000",  # Base year employment
            f"EUUR01{soc_clean}00000000",  # Projected year employment
        ]
        
    # BLS projections are also available as flat files
    # https://www.bls.gov/emp/tables.htm
    # For now, we'll note which occupations have growth projections
    
    proj_url = "https://data.bls.gov/projections/occupationProj"
    print(f"  Note: Full projections available at {proj_url}")
    print(f"  Will integrate in next iteration")
    
    return projections


def main():
    # Collect OES employment and wage data
    oes_data = collect_oes_data()
    
    # Collect projections
    collect_bls_projections()
    
    print(f"\n{'='*70}")
    print(f"  COLLECTION COMPLETE")
    print(f"{'='*70}")
    print(f"  SOC codes collected: {len(oes_data)}")


if __name__ == "__main__":
    main()
