"""
Fix Workday API URLs and retry failed companies.
Many Workday sites changed their URL structure — this tries common patterns.
Also adds Visa (SmartRecruiters), Microsoft, and other custom scrapers.
"""
import requests
import json
import os
import time
from datetime import datetime

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'market_signals', 'career_pages')
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Content-Type": "application/json",
}

# Companies that failed — try alternative URLs
RETRY_CONFIGS = {
    "John Deere": [
        "https://johndeere.wd1.myworkdayjobs.com/wday/cxs/johndeere/ExternalJobs/jobs",
        "https://johndeere.wd5.myworkdayjobs.com/wday/cxs/johndeere/External/jobs",
    ],
    "Dell": [
        "https://dell.wd1.myworkdayjobs.com/wday/cxs/dell/ExternalNonSales/jobs",
        "https://dell.wd5.myworkdayjobs.com/wday/cxs/dell/External/jobs",
    ],
    "GAP": [
        "https://gapinc.wd5.myworkdayjobs.com/wday/cxs/gapinc/careers/jobs",
        "https://gapinc.wd5.myworkdayjobs.com/wday/cxs/gapinc/GapIncCareers/jobs",
        "https://www.gapinc.com/api/careers/jobs",
    ],
    "Novo Nordisk": [
        "https://novonordisk.wd1.myworkdayjobs.com/wday/cxs/novonordisk/Careers/jobs",
        "https://novonordisk.wd5.myworkdayjobs.com/wday/cxs/novonordisk/Search/jobs",
    ],
    "GE Aerospace": [
        "https://geaerospace.wd5.myworkdayjobs.com/wday/cxs/geaerospace/GEAerospace_Careers/jobs",
        "https://ge.wd5.myworkdayjobs.com/wday/cxs/ge/GE_Aerospace/jobs",
    ],
    "GE Healthcare": [
        "https://gehealthcare.wd5.myworkdayjobs.com/wday/cxs/gehealthcare/GEHealthCareCareers/jobs",
        "https://gehealthcare.wd5.myworkdayjobs.com/wday/cxs/gehealthcare/careers/jobs",
    ],
    "SAIC": [
        "https://saic.wd1.myworkdayjobs.com/wday/cxs/saic/SAIC_Careers/jobs",
        "https://rolp.co/api/saic/jobs",
    ],
    "Solidigm": [
        "https://solidigm.wd1.myworkdayjobs.com/wday/cxs/solidigm/Solidigm_Careers/jobs",
        "https://solidigm.wd5.myworkdayjobs.com/wday/cxs/solidigm/Solidigm/jobs",
    ],
    "Zebra Technologies": [
        "https://zebra.wd1.myworkdayjobs.com/wday/cxs/zebra/Zebra_Careers/jobs",
        "https://zebra.wd5.myworkdayjobs.com/wday/cxs/zebra/Careers/jobs",
    ],
    "Mercedes-Benz": [
        "https://mercedesbenz.wd3.myworkdayjobs.com/wday/cxs/mercedesbenz/MBCareers/jobs",
        "https://daimler.wd3.myworkdayjobs.com/wday/cxs/daimler/Daimler_Careers/jobs",
    ],
    "Daimler": [
        "https://daimler.wd3.myworkdayjobs.com/wday/cxs/daimler/Daimler_Careers/jobs",
        "https://daimlertruck.wd3.myworkdayjobs.com/wday/cxs/daimlertruck/daimler_jobs/jobs",
    ],
    "Expedia": [
        "https://expedia.wd5.myworkdayjobs.com/wday/cxs/expedia/Expedia_Careers/jobs",
        "https://lifeatexpediagroup.com/api/jobs",
    ],
    "Integra LifeSciences": [
        "https://integralife.wd1.myworkdayjobs.com/wday/cxs/integralife/Integra_Careers/jobs",
        "https://integralife.wd5.myworkdayjobs.com/wday/cxs/integralife/IntegraCareers/jobs",
    ],
}

# Additional companies to try with known APIs
ADDITIONAL = {
    "Visa": {
        "type": "smartrecruiters",
        "api": "https://api.smartrecruiters.com/v1/companies/Visa/postings",
    },
    "Microsoft": {
        "type": "microsoft",
        "api": "https://gcsservices.careers.microsoft.com/search/api/v1/search",
    },
    "Red Hat": {
        "type": "greenhouse",
        "api": "https://boards-api.greenhouse.io/v1/boards/redhat/jobs",
    },
    "Blue Origin": {
        "type": "greenhouse",
        "api": "https://boards-api.greenhouse.io/v1/boards/blueorigin/jobs",
    },
}


def try_workday(url):
    """Try a single Workday API URL."""
    payload = {"appliedFacets": {}, "limit": 20, "offset": 0, "searchText": ""}
    try:
        resp = requests.post(url, json=payload, headers=HEADERS, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            jobs = data.get("jobPostings", [])
            total = data.get("total", 0)
            return jobs, total
    except:
        pass
    return None, 0


def fetch_all_workday(url, max_jobs=200):
    """Fetch all jobs from a working Workday URL."""
    jobs = []
    offset = 0
    while offset < max_jobs:
        payload = {"appliedFacets": {}, "limit": 20, "offset": offset, "searchText": ""}
        try:
            resp = requests.post(url, json=payload, headers=HEADERS, timeout=10)
            if resp.status_code != 200:
                break
            data = resp.json()
            batch = data.get("jobPostings", [])
            if not batch:
                break
            for jp in batch:
                jobs.append({
                    "title": jp.get("title", ""),
                    "location": jp.get("locationsText", ""),
                    "posted_date": jp.get("postedOn", ""),
                })
            total = data.get("total", 0)
            offset += 20
            if offset >= total:
                break
            time.sleep(0.3)
        except:
            break
    return jobs


def fetch_greenhouse(api_url):
    """Fetch from Greenhouse API."""
    try:
        resp = requests.get(api_url, headers={"Accept": "application/json"}, timeout=10)
        if resp.status_code != 200:
            return []
        data = resp.json()
        jobs_list = data.get("jobs", data) if isinstance(data, dict) else data
        jobs = []
        for jp in jobs_list[:200]:
            loc = jp.get("location", {})
            loc_name = loc.get("name", "") if isinstance(loc, dict) else str(loc)
            jobs.append({
                "title": jp.get("title", ""),
                "location": loc_name,
                "posted_date": jp.get("updated_at", ""),
            })
        return jobs
    except:
        return []


def fetch_smartrecruiters(api_url):
    """Fetch from SmartRecruiters API."""
    jobs = []
    offset = 0
    while offset < 500:
        try:
            resp = requests.get(api_url, params={"offset": offset, "limit": 100}, timeout=10)
            if resp.status_code != 200:
                break
            content = resp.json().get("content", [])
            if not content:
                break
            for jp in content:
                loc = jp.get("location", {})
                jobs.append({
                    "title": jp.get("name", ""),
                    "location": f"{loc.get('city', '')}, {loc.get('region', '')}".strip(", "),
                    "posted_date": jp.get("releasedDate", ""),
                })
            offset += 100
            time.sleep(0.3)
        except:
            break
    return jobs


def fetch_microsoft(api_url):
    """Fetch from Microsoft careers API."""
    jobs = []
    try:
        params = {"lc": "United States", "l": "en_us", "pg": 1, "pgSz": 100, "o": "Recent"}
        resp = requests.get(api_url, params=params, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            for jp in data.get("operationResult", {}).get("result", {}).get("jobs", []):
                jobs.append({
                    "title": jp.get("title", ""),
                    "location": jp.get("properties", {}).get("primaryLocation", ""),
                    "posted_date": jp.get("properties", {}).get("postedDate", ""),
                })
    except Exception as e:
        print(f"    Error: {e}")
    return jobs


def main():
    print("=" * 70)
    print("  RETRY FAILED COMPANIES + ADDITIONAL SCRAPERS")
    print("=" * 70)
    
    all_results = {}
    
    # Retry Workday companies
    for company, urls in RETRY_CONFIGS.items():
        print(f"\n  [{company}] Trying {len(urls)} URLs...")
        found = False
        for url in urls:
            print(f"    → {url.split('/cxs/')[-1] if '/cxs/' in url else url[:60]}...", end=" ")
            batch, total = try_workday(url)
            if batch:
                print(f"✓ ({total} total)")
                jobs = fetch_all_workday(url)
                all_results[company] = {"jobs": jobs, "total": len(jobs), "api_url": url}
                found = True
                break
            else:
                print("✗")
        if not found:
            print(f"    ❌ No working URL found")
    
    # Additional companies
    print(f"\n{'='*70}")
    print(f"  ADDITIONAL COMPANIES")
    print(f"{'='*70}")
    
    for company, config in ADDITIONAL.items():
        print(f"\n  [{company}] ({config['type']})...", end=" ")
        
        if config["type"] == "greenhouse":
            jobs = fetch_greenhouse(config["api"])
        elif config["type"] == "smartrecruiters":
            jobs = fetch_smartrecruiters(config["api"])
        elif config["type"] == "microsoft":
            jobs = fetch_microsoft(config["api"])
        else:
            jobs = []
        
        if jobs:
            all_results[company] = {"jobs": jobs, "total": len(jobs), "api_url": config["api"]}
            print(f"✓ {len(jobs)} jobs")
        else:
            print(f"✗ 0 jobs")
        
        time.sleep(1)
    
    # Save results
    for company, result in all_results.items():
        safe_name = company.lower().replace(" ", "_").replace("/", "_")
        
        # Extract skill signals from titles
        skill_keywords = {
            "python": "Python", "java": "Java", "react": "React", "angular": "Angular",
            "aws": "AWS", "azure": "Azure", "gcp": "GCP", "kubernetes": "Kubernetes",
            "docker": "Docker", "sql": "SQL", "machine learning": "Machine Learning",
            "ai": "AI", "devops": "DevOps", "cloud": "Cloud", "security": "Security",
            ".net": ".NET", "salesforce": "Salesforce", "sap": "SAP",
        }
        skills = {}
        title_counts = {}
        loc_counts = {}
        for j in result["jobs"]:
            t = j.get("title", "")
            title_counts[t] = title_counts.get(t, 0) + 1
            loc = j.get("location", "")
            loc_counts[loc] = loc_counts.get(loc, 0) + 1
            for kw, skill in skill_keywords.items():
                if kw in t.lower():
                    skills[skill] = skills.get(skill, 0) + 1
        
        output = {
            "company": company,
            "total_openings": result["total"],
            "scraped_at": datetime.now().isoformat(),
            "api_url": result["api_url"],
            "top_titles": dict(sorted(title_counts.items(), key=lambda x: x[1], reverse=True)[:20]),
            "top_locations": dict(sorted(loc_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
            "skill_signals": dict(sorted(skills.items(), key=lambda x: x[1], reverse=True)),
            "jobs": result["jobs"],
        }
        
        path = os.path.join(OUTPUT_DIR, f"{safe_name}.json")
        with open(path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Saved: {path}")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  RETRY RESULTS")
    print(f"{'='*70}")
    total = sum(r["total"] for r in all_results.values())
    print(f"  New companies scraped: {len(all_results)}")
    print(f"  New jobs found: {total}")
    for company, result in sorted(all_results.items(), key=lambda x: x[1]["total"], reverse=True):
        print(f"    {company:<30} {result['total']:>5} jobs")


if __name__ == "__main__":
    main()
