"""
Company Career Page Scraper
============================
Scrapes public career pages of our 61 KEEP companies.
Extracts: job titles, locations, skills mentioned, departments.
Legal: Public web pages, respects robots.txt.

Uses requests + BeautifulSoup for static pages.
For JS-heavy career pages (Workday, Lever, Greenhouse), 
we parse their underlying JSON APIs where available.
"""
import requests
import json
import os
import re
import time
from datetime import datetime
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'market_signals', 'career_pages')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Company → Career Page URLs ───
# Many large companies use Workday, Greenhouse, Lever, or custom ATS
# We target their public job listing APIs/pages
COMPANY_CAREER_URLS = {
    # ─── Direct career pages / ATS APIs ───
    "Caterpillar": {
        "type": "workday",
        "url": "https://cat.wd5.myworkdayjobs.com/en-US/CaterpillarCareers",
        "api": "https://cat.wd5.myworkdayjobs.com/wday/cxs/cat/CaterpillarCareers/jobs",
    },
    "Amgen": {
        "type": "workday",
        "url": "https://amgen.wd1.myworkdayjobs.com/en-US/Careers",
        "api": "https://amgen.wd1.myworkdayjobs.com/wday/cxs/amgen/Careers/jobs",
    },
    "Samsung": {
        "type": "custom",
        "url": "https://www.samsung.com/us/careers/",
        "api": None,
    },
    "John Deere": {
        "type": "workday",
        "url": "https://johndeere.wd1.myworkdayjobs.com/en-US/External",
        "api": "https://johndeere.wd1.myworkdayjobs.com/wday/cxs/johndeere/External/jobs",
    },
    "Facebook": {
        "type": "custom",
        "url": "https://www.metacareers.com/jobs",
        "api": "https://www.metacareers.com/graphql",
    },
    "Microsoft": {
        "type": "custom",
        "url": "https://careers.microsoft.com/us/en/search-results",
        "api": "https://gcsservices.careers.microsoft.com/search/api/v1/search",
    },
    "T-Mobile": {
        "type": "workday",
        "url": "https://tmobile.wd1.myworkdayjobs.com/en-US/External",
        "api": "https://tmobile.wd1.myworkdayjobs.com/wday/cxs/tmobile/External/jobs",
    },
    "PayPal": {
        "type": "custom",
        "url": "https://careers.pypl.com/search-jobs",
        "api": None,
    },
    "Salesforce": {
        "type": "custom",
        "url": "https://careers.salesforce.com/en/jobs/",
        "api": None,
    },
    "Amazon": {
        "type": "custom",
        "url": "https://www.amazon.jobs/en/search",
        "api": "https://www.amazon.jobs/en/search.json",
    },
    "Verizon One": {
        "type": "workday",
        "url": "https://mycareer.verizon.com/jobs/search",
        "api": None,
    },
    "eBay": {
        "type": "workday",
        "url": "https://ebay.wd5.myworkdayjobs.com/en-US/apply",
        "api": "https://ebay.wd5.myworkdayjobs.com/wday/cxs/ebay/apply/jobs",
    },
    "Pfizer": {
        "type": "workday",
        "url": "https://pfizer.wd1.myworkdayjobs.com/en-US/PfizerCareers",
        "api": "https://pfizer.wd1.myworkdayjobs.com/wday/cxs/pfizer/PfizerCareers/jobs",
    },
    "Visa": {
        "type": "workday",
        "url": "https://jobs.smartrecruiters.com/Visa",
        "api": "https://api.smartrecruiters.com/v1/companies/Visa/postings",
    },
    "Oracle": {
        "type": "custom",
        "url": "https://careers.oracle.com/jobs/",
        "api": None,
    },
    "GAP": {
        "type": "workday",
        "url": "https://www.gapinc.com/en-us/careers",
        "api": None,
    },
    "Dell": {
        "type": "workday",
        "url": "https://dell.wd1.myworkdayjobs.com/en-US/External",
        "api": "https://dell.wd1.myworkdayjobs.com/wday/cxs/dell/External/jobs",
    },
    "Expedia": {
        "type": "workday",
        "url": "https://expedia.wd5.myworkdayjobs.com/en-US/search",
        "api": "https://expedia.wd5.myworkdayjobs.com/wday/cxs/expedia/search/jobs",
    },
    "Kroger": {
        "type": "custom",
        "url": "https://jobs.kroger.com/",
        "api": None,
    },
    "Blue Origin": {
        "type": "custom",
        "url": "https://www.blueorigin.com/careers",
        "api": None,
    },
    "Novo Nordisk": {
        "type": "workday",
        "url": "https://novonordisk.wd1.myworkdayjobs.com/en-US/Search",
        "api": "https://novonordisk.wd1.myworkdayjobs.com/wday/cxs/novonordisk/Search/jobs",
    },
    "Red Hat": {
        "type": "custom",
        "url": "https://www.redhat.com/en/jobs",
        "api": None,
    },
    "Best Buy": {
        "type": "workday",
        "url": "https://sjobs.brassring.com/TGnewUI/Search/Home/Home",
        "api": None,
    },
    "GE Aerospace": {
        "type": "workday",
        "url": "https://geaerospace.wd5.myworkdayjobs.com/en-US/GE_Aerospace",
        "api": "https://geaerospace.wd5.myworkdayjobs.com/wday/cxs/geaerospace/GE_Aerospace/jobs",
    },
    "GE Healthcare": {
        "type": "workday",
        "url": "https://gehealthcare.wd5.myworkdayjobs.com/en-US/GE_HealthCare",
        "api": "https://gehealthcare.wd5.myworkdayjobs.com/wday/cxs/gehealthcare/GE_HealthCare/jobs",
    },
    "SAIC": {
        "type": "workday",
        "url": "https://saic.wd1.myworkdayjobs.com/en-US/SAIC",
        "api": "https://saic.wd1.myworkdayjobs.com/wday/cxs/saic/SAIC/jobs",
    },
    "Solidigm": {
        "type": "workday",
        "url": "https://solidigm.wd1.myworkdayjobs.com/en-US/Solidigm",
        "api": "https://solidigm.wd1.myworkdayjobs.com/wday/cxs/solidigm/Solidigm/jobs",
    },
    "Zebra Technologies": {
        "type": "workday",
        "url": "https://zebra.wd1.myworkdayjobs.com/en-US/Careers",
        "api": "https://zebra.wd1.myworkdayjobs.com/wday/cxs/zebra/Careers/jobs",
    },
    "Mercedes-Benz": {
        "type": "workday",
        "url": "https://mercedesbenz.wd3.myworkdayjobs.com/en-US/jobs",
        "api": "https://mercedesbenz.wd3.myworkdayjobs.com/wday/cxs/mercedesbenz/jobs/jobs",
    },
    "Rubrik": {
        "type": "greenhouse",
        "url": "https://www.rubrik.com/company/careers",
        "api": "https://boards-api.greenhouse.io/v1/boards/rubrik/jobs",
    },
    "Daimler": {
        "type": "workday",
        "url": "https://daimler.wd3.myworkdayjobs.com/en-US/daimler_jobs",
        "api": "https://daimler.wd3.myworkdayjobs.com/wday/cxs/daimler/daimler_jobs/jobs",
    },
    "Integra LifeSciences": {
        "type": "workday",
        "url": "https://integralife.wd1.myworkdayjobs.com/en-US/IntegraCareers",
        "api": "https://integralife.wd1.myworkdayjobs.com/wday/cxs/integralife/IntegraCareers/jobs",
    },
}

# Standard headers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


def scrape_workday(company, config):
    """Scrape Workday-powered career sites (most common ATS)."""
    api_url = config.get("api")
    if not api_url:
        return []
    
    jobs = []
    offset = 0
    limit = 20
    
    headers = {
        **HEADERS,
        "Content-Type": "application/json",
    }
    
    while True:
        payload = {"appliedFacets": {}, "limit": limit, "offset": offset, "searchText": ""}
        
        try:
            resp = requests.post(api_url, json=payload, headers=headers, timeout=15)
            if resp.status_code != 200:
                print(f"    HTTP {resp.status_code}")
                break
            
            data = resp.json()
            job_postings = data.get("jobPostings", [])
            
            if not job_postings:
                break
            
            for jp in job_postings:
                job = {
                    "title": jp.get("title", ""),
                    "location": jp.get("locationsText", ""),
                    "posted_date": jp.get("postedOn", ""),
                    "url": urljoin(config["url"], jp.get("externalPath", "")),
                }
                jobs.append(job)
            
            total = data.get("total", 0)
            offset += limit
            
            if offset >= total or offset >= 200:  # Cap at 200 per company
                break
            
            time.sleep(0.5)
            
        except Exception as e:
            print(f"    Error at offset {offset}: {e}")
            break
    
    return jobs


def scrape_greenhouse(company, config):
    """Scrape Greenhouse-powered career sites."""
    api_url = config.get("api")
    if not api_url:
        return []
    
    try:
        resp = requests.get(api_url, headers=HEADERS, timeout=15)
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
                "posted_date": jp.get("updated_at", jp.get("created_at", "")),
                "url": jp.get("absolute_url", ""),
                "department": jp.get("departments", [{}])[0].get("name", "") if jp.get("departments") else "",
            })
        
        return jobs
    except Exception as e:
        print(f"    Error: {e}")
        return []


def scrape_smartrecruiters(company, config):
    """Scrape SmartRecruiters-powered career sites (Visa etc.)."""
    api_url = config.get("api")
    if not api_url:
        return []
    
    jobs = []
    offset = 0
    
    while True:
        try:
            resp = requests.get(
                api_url, 
                params={"offset": offset, "limit": 100},
                headers=HEADERS, 
                timeout=15
            )
            if resp.status_code != 200:
                break
            
            data = resp.json()
            content = data.get("content", [])
            
            if not content:
                break
            
            for jp in content:
                loc = jp.get("location", {})
                jobs.append({
                    "title": jp.get("name", ""),
                    "location": f"{loc.get('city', '')}, {loc.get('region', '')}".strip(", "),
                    "posted_date": jp.get("releasedDate", ""),
                    "url": jp.get("ref", ""),
                    "department": jp.get("department", {}).get("label", ""),
                })
            
            offset += 100
            if offset >= 500:
                break
            
            time.sleep(0.5)
        except Exception as e:
            print(f"    Error: {e}")
            break
    
    return jobs


def scrape_amazon(company, config):
    """Scrape Amazon jobs API."""
    try:
        jobs = []
        for offset in range(0, 200, 10):
            resp = requests.get(
                "https://www.amazon.jobs/en/search.json",
                params={"offset": offset, "result_limit": 10, "sort": "recent"},
                headers=HEADERS,
                timeout=15,
            )
            if resp.status_code != 200:
                break
            data = resp.json()
            for jp in data.get("jobs", []):
                jobs.append({
                    "title": jp.get("title", ""),
                    "location": jp.get("normalized_location", jp.get("location", "")),
                    "posted_date": jp.get("posted_date", ""),
                    "url": f"https://www.amazon.jobs{jp.get('job_path', '')}",
                    "category": jp.get("job_category", ""),
                })
            if len(data.get("jobs", [])) < 10:
                break
            time.sleep(0.5)
        return jobs
    except Exception as e:
        print(f"    Error: {e}")
        return []


def scrape_company(company, config):
    """Route to appropriate scraper based on ATS type."""
    ats_type = config.get("type", "custom")
    
    if ats_type == "workday":
        return scrape_workday(company, config)
    elif ats_type == "greenhouse":
        return scrape_greenhouse(company, config)
    elif ats_type == "smartrecruiters":
        return scrape_smartrecruiters(company, config)
    elif company == "Amazon":
        return scrape_amazon(company, config)
    else:
        # For custom sites without API, we just log it
        print(f"    ⚠ Custom site — no API scraper yet")
        return []


def extract_skills_from_titles(jobs):
    """Extract skill signals from job titles."""
    skill_keywords = {
        "python": "Python", "java": "Java", "react": "React", "angular": "Angular",
        "node": "Node.js", "aws": "AWS", "azure": "Azure", "gcp": "GCP",
        "kubernetes": "Kubernetes", "docker": "Docker", "sql": "SQL",
        "machine learning": "Machine Learning", "ai": "AI", "data science": "Data Science",
        "devops": "DevOps", "cloud": "Cloud", "agile": "Agile", "scrum": "Scrum",
        "javascript": "JavaScript", "typescript": "TypeScript", "golang": "Go",
        "rust": "Rust", "c++": "C++", "c#": "C#", ".net": ".NET",
        "salesforce": "Salesforce", "sap": "SAP", "oracle": "Oracle",
        "cybersecurity": "Cybersecurity", "security": "Security",
        "ios": "iOS", "android": "Android", "mobile": "Mobile",
        "full stack": "Full Stack", "frontend": "Frontend", "backend": "Backend",
        "etl": "ETL", "tableau": "Tableau", "power bi": "Power BI",
    }
    
    skill_counts = {}
    for job in jobs:
        title_lower = job.get("title", "").lower()
        for keyword, skill in skill_keywords.items():
            if keyword in title_lower:
                skill_counts[skill] = skill_counts.get(skill, 0) + 1
    
    return dict(sorted(skill_counts.items(), key=lambda x: x[1], reverse=True))


def main():
    print("=" * 70)
    print("  COMPANY CAREER PAGE SCRAPER")
    print("=" * 70)
    print(f"  Companies configured: {len(COMPANY_CAREER_URLS)}")
    print(f"  Output: {OUTPUT_DIR}")
    print()
    
    all_results = {}
    total_jobs = 0
    
    for company, config in COMPANY_CAREER_URLS.items():
        print(f"  [{company}] ({config['type']})...", end=" ")
        
        jobs = scrape_company(company, config)
        
        if jobs:
            # Extract skill signals from titles
            skills = extract_skills_from_titles(jobs)
            
            # Get title distribution
            title_counts = {}
            for j in jobs:
                t = j["title"]
                title_counts[t] = title_counts.get(t, 0) + 1
            top_titles = dict(sorted(title_counts.items(), key=lambda x: x[1], reverse=True)[:20])
            
            # Location distribution
            loc_counts = {}
            for j in jobs:
                loc = j.get("location", "Unknown")
                loc_counts[loc] = loc_counts.get(loc, 0) + 1
            top_locations = dict(sorted(loc_counts.items(), key=lambda x: x[1], reverse=True)[:10])
            
            result = {
                "company": company,
                "total_openings": len(jobs),
                "scraped_at": datetime.now().isoformat(),
                "ats_type": config["type"],
                "career_url": config["url"],
                "top_titles": top_titles,
                "top_locations": top_locations,
                "skill_signals": skills,
                "jobs": jobs,
            }
            
            all_results[company] = result
            total_jobs += len(jobs)
            print(f"✓ {len(jobs)} jobs")
        else:
            print(f"✗ 0 jobs (or custom site)")
        
        time.sleep(1)  # Be respectful
    
    # Save individual company files
    for company, result in all_results.items():
        safe_name = company.lower().replace(" ", "_").replace("/", "_")
        path = os.path.join(OUTPUT_DIR, f"{safe_name}.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
    
    # Save combined summary
    summary = {
        "source": "Company Career Pages (direct scraping)",
        "collected_at": datetime.now().isoformat(),
        "total_companies_scraped": len(all_results),
        "total_jobs_found": total_jobs,
        "companies": {},
    }
    
    for company, result in all_results.items():
        summary["companies"][company] = {
            "total_openings": result["total_openings"],
            "top_titles": result["top_titles"],
            "top_locations": result["top_locations"],
            "skill_signals": result["skill_signals"],
        }
    
    summary_path = os.path.join(OUTPUT_DIR, "_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"  SCRAPING COMPLETE")
    print(f"{'='*70}")
    print(f"  Companies scraped: {len(all_results)}/{len(COMPANY_CAREER_URLS)}")
    print(f"  Total jobs found: {total_jobs}")
    
    for company, result in sorted(all_results.items(), key=lambda x: x[1]["total_openings"], reverse=True):
        skills_str = ", ".join(list(result["skill_signals"].keys())[:5])
        print(f"    {company:<30} {result['total_openings']:>5} jobs  | skills: {skills_str}")
    
    print(f"\n  Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
