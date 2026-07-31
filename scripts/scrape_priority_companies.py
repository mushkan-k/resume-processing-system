"""
Targeted Career Page Scraper — 10 companies that need more data.
GAP, RELX, Campbells, PepsiCo, ADVARRA, Diageo, Seqirus, Daimler, Mercedes-Benz, GE Healthcare.

Strategy per company:
- Check Workday, Greenhouse, Lever, SmartRecruiters, iCIMS APIs
- Try direct career page HTML parsing as fallback
- For each, we discover the right ATS URL first
"""
import requests
import json
import os
import re
import time
from datetime import datetime
from bs4 import BeautifulSoup
from urllib.parse import urljoin

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'market_signals', 'career_pages')
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "application/json, text/html, */*",
    "Accept-Language": "en-US,en;q=0.9",
}

SKILL_KEYWORDS = {
    "python": "Python", "java ": "Java", "react": "React", "angular": "Angular",
    "node.js": "Node.js", "nodejs": "Node.js", "aws": "AWS", "azure": "Azure",
    "gcp": "GCP", "kubernetes": "Kubernetes", "docker": "Docker", "sql": "SQL",
    "machine learning": "Machine Learning", "ai": "AI", "artificial intelligence": "AI",
    "data science": "Data Science", "devops": "DevOps", "cloud": "Cloud",
    "agile": "Agile", "scrum": "Scrum", "javascript": "JavaScript",
    "typescript": "TypeScript", "golang": "Go", "rust": "Rust", "c++": "C++",
    "c#": "C#", ".net": ".NET", "salesforce": "Salesforce", "sap": "SAP",
    "oracle": "Oracle", "cybersecurity": "Cybersecurity", "security": "Security",
    "ios": "iOS", "android": "Android", "mobile": "Mobile", "full stack": "Full Stack",
    "frontend": "Frontend", "backend": "Backend", "etl": "ETL", "tableau": "Tableau",
    "power bi": "Power BI", "terraform": "Terraform", "jenkins": "Jenkins",
    "jira": "Jira", "excel": "Excel", "pharma": "Pharma", "clinical": "Clinical",
    "supply chain": "Supply Chain", "manufacturing": "Manufacturing",
    "quality": "Quality", "regulatory": "Regulatory", "compliance": "Compliance",
}


# ════════════════════════════════════════════════════════════════
# ATS-specific scrapers
# ════════════════════════════════════════════════════════════════

def scrape_workday(api_url, max_jobs=300):
    """Scrape Workday ATS API."""
    jobs = []
    offset = 0
    while offset < max_jobs:
        payload = {"appliedFacets": {}, "limit": 20, "offset": offset, "searchText": ""}
        try:
            resp = requests.post(api_url, json=payload,
                                 headers={**HEADERS, "Content-Type": "application/json"}, timeout=15)
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
            if offset + 20 >= data.get("total", 0):
                break
            offset += 20
            time.sleep(0.3)
        except Exception as e:
            print(f"      Workday error at offset {offset}: {e}")
            break
    return jobs


def scrape_greenhouse(board_token, max_jobs=300):
    """Scrape Greenhouse boards API."""
    url = f"https://boards-api.greenhouse.io/v1/boards/{board_token}/jobs"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return []
        data = resp.json()
        jobs_list = data.get("jobs", []) if isinstance(data, dict) else data
        jobs = []
        for jp in jobs_list[:max_jobs]:
            loc = jp.get("location", {})
            loc_name = loc.get("name", "") if isinstance(loc, dict) else str(loc)
            depts = jp.get("departments", [])
            dept = depts[0].get("name", "") if depts else ""
            jobs.append({
                "title": jp.get("title", ""),
                "location": loc_name,
                "posted_date": jp.get("updated_at", ""),
                "department": dept,
            })
        return jobs
    except Exception as e:
        print(f"      Greenhouse error: {e}")
        return []


def scrape_lever(company_slug, max_jobs=300):
    """Scrape Lever postings API."""
    url = f"https://api.lever.co/v0/postings/{company_slug}"
    try:
        resp = requests.get(url, params={"mode": "json", "limit": max_jobs}, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            return []
        data = resp.json()
        jobs = []
        for jp in data[:max_jobs]:
            cats = jp.get("categories", {})
            jobs.append({
                "title": jp.get("text", ""),
                "location": cats.get("location", ""),
                "department": cats.get("department", ""),
                "team": cats.get("team", ""),
                "posted_date": "",
            })
        return jobs
    except Exception as e:
        print(f"      Lever error: {e}")
        return []


def scrape_smartrecruiters(company_id, max_jobs=300):
    """Scrape SmartRecruiters API."""
    url = f"https://api.smartrecruiters.com/v1/companies/{company_id}/postings"
    jobs = []
    offset = 0
    while offset < max_jobs:
        try:
            resp = requests.get(url, params={"offset": offset, "limit": 100}, headers=HEADERS, timeout=15)
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
                    "department": jp.get("department", {}).get("label", ""),
                    "posted_date": jp.get("releasedDate", ""),
                })
            offset += 100
            time.sleep(0.3)
        except Exception as e:
            print(f"      SmartRecruiters error: {e}")
            break
    return jobs


def scrape_icims(company_url, max_jobs=300):
    """Scrape iCIMS career portal (HTML parsing)."""
    jobs = []
    try:
        resp = requests.get(company_url, headers={**HEADERS, "Accept": "text/html"}, timeout=15)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        # iCIMS uses various class names
        for link in soup.select("a.iCIMS_Anchor, a[class*='job'], div.iCIMS_JobsTable a"):
            title = link.get_text(strip=True)
            href = link.get("href", "")
            if title and len(title) > 3:
                jobs.append({"title": title, "location": "", "posted_date": "", "url": href})
        return jobs[:max_jobs]
    except Exception as e:
        print(f"      iCIMS error: {e}")
        return []


def scrape_html_career_page(url, max_jobs=300):
    """Generic HTML scraper for career pages."""
    jobs = []
    try:
        resp = requests.get(url, headers={**HEADERS, "Accept": "text/html"}, timeout=15, allow_redirects=True)
        if resp.status_code != 200:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")

        # Strategy 1: look for job listing links with common patterns
        job_links = soup.select(
            'a[href*="job"], a[href*="career"], a[href*="position"], '
            'a[href*="opening"], a[href*="apply"], '
            'a.job-title, a.posting-title, h2 a, h3 a, '
            'div[class*="job"] a, li[class*="job"] a'
        )
        seen_titles = set()
        for link in job_links:
            title = link.get_text(strip=True)
            if title and len(title) > 5 and len(title) < 150 and title not in seen_titles:
                seen_titles.add(title)
                # Try to find location nearby
                parent = link.find_parent(['li', 'div', 'tr'])
                location = ""
                if parent:
                    loc_el = parent.select_one('[class*="location"], [class*="Location"], span.location')
                    if loc_el:
                        location = loc_el.get_text(strip=True)
                jobs.append({"title": title, "location": location, "posted_date": ""})

        # Strategy 2: look for structured data (JSON-LD)
        for script in soup.select('script[type="application/ld+json"]'):
            try:
                ld = json.loads(script.string)
                if isinstance(ld, list):
                    for item in ld:
                        if item.get("@type") == "JobPosting":
                            jobs.append({
                                "title": item.get("title", ""),
                                "location": str(item.get("jobLocation", "")),
                                "posted_date": item.get("datePosted", ""),
                            })
                elif isinstance(ld, dict) and ld.get("@type") == "JobPosting":
                    jobs.append({
                        "title": ld.get("title", ""),
                        "location": str(ld.get("jobLocation", "")),
                        "posted_date": ld.get("datePosted", ""),
                    })
            except:
                pass

        return jobs[:max_jobs]
    except Exception as e:
        print(f"      HTML scrape error: {e}")
        return []


# ════════════════════════════════════════════════════════════════
# ATS discovery — try multiple ATS platforms per company
# ════════════════════════════════════════════════════════════════

def discover_workday_url(company_slug, subdomain_options):
    """Try multiple Workday URL patterns."""
    for sub in subdomain_options:
        for wd_ver in ["wd1", "wd3", "wd5"]:
            url = f"https://{sub}.{wd_ver}.myworkdayjobs.com/wday/cxs/{sub}/{company_slug}/jobs"
            try:
                resp = requests.post(url, json={"appliedFacets": {}, "limit": 1, "offset": 0, "searchText": ""},
                                     headers={**HEADERS, "Content-Type": "application/json"}, timeout=8)
                if resp.status_code == 200:
                    data = resp.json()
                    total = data.get("total", 0)
                    if total > 0:
                        return url, total
            except:
                pass
    return None, 0


# ════════════════════════════════════════════════════════════════
# Company-specific configurations
# ════════════════════════════════════════════════════════════════

COMPANIES = {
    "GAP": {
        "strategies": [
            ("workday_discover", {"slugs": ["gapinc", "gap"], "sites": [
                "careers", "Careers", "GapIncCareers", "External", "gapinc_careers",
                "Gap_Careers", "Gap", "GapCareers"
            ]}),
            ("greenhouse", "gapinc"),
            ("greenhouse", "gap"),
            ("lever", "gap-inc"),
            ("lever", "gapinc"),
            ("html", "https://www.gapinc.com/en-us/careers"),
        ],
    },
    "RELX Inc.": {
        "strategies": [
            ("workday_discover", {"slugs": ["relx", "relxgroup"], "sites": [
                "External", "Careers", "RELX_Careers", "RELXCareers"
            ]}),
            ("greenhouse", "relx"),
            ("greenhouse", "relxgroup"),
            ("smartrecruiters", "RELX"),
            ("smartrecruiters", "RELXGroup"),
            ("html", "https://www.relx.com/careers"),
            ("html", "https://careers.relx.com"),
        ],
    },
    "Campbells": {
        "strategies": [
            ("workday_discover", {"slugs": ["campbells", "campbellsoup", "campbell"], "sites": [
                "External", "Careers", "CampbellCareers", "Campbell_Careers"
            ]}),
            ("greenhouse", "campbells"),
            ("greenhouse", "campbellsoupcompany"),
            ("smartrecruiters", "CampbellSoupCompany"),
            ("html", "https://careers.campbellsoupcompany.com"),
            ("html", "https://www.campbellsoupcompany.com/careers/"),
        ],
    },
    "PepsiCo Inc": {
        "strategies": [
            ("workday_discover", {"slugs": ["pepsico"], "sites": [
                "External", "Careers", "PepsiCo_Careers", "PepsiCoCareers", "pepsicojobs"
            ]}),
            ("greenhouse", "pepsico"),
            ("html", "https://www.pepsicojobs.com/main/jobs"),
            ("html", "https://www.pepsico.com/our-story/careers"),
        ],
    },
    "ADVARRA, Inc.": {
        "strategies": [
            ("greenhouse", "advarra"),
            ("lever", "advarra"),
            ("workday_discover", {"slugs": ["advarra"], "sites": [
                "External", "Careers", "Advarra_Careers"
            ]}),
            ("smartrecruiters", "Advarra"),
            ("html", "https://www.advarra.com/careers/"),
        ],
    },
    "Diageo": {
        "strategies": [
            ("workday_discover", {"slugs": ["diageo"], "sites": [
                "External", "Careers", "DiageoCareers", "Diageo_Careers"
            ]}),
            ("smartrecruiters", "Diageo"),
            ("greenhouse", "diageo"),
            ("html", "https://careers.diageo.com"),
            ("html", "https://www.diageo.com/en/careers"),
        ],
    },
    "Seqirus": {
        "strategies": [
            ("workday_discover", {"slugs": ["seqirus", "csl"], "sites": [
                "External", "Careers", "CSL_Careers", "SeqirusCareers"
            ]}),
            ("greenhouse", "seqirus"),
            ("lever", "seqirus"),
            ("smartrecruiters", "Seqirus"),
            ("html", "https://www.seqirus.com/careers"),
            ("html", "https://careers.csl.com"),
        ],
    },
    "Daimler": {
        "strategies": [
            ("workday_discover", {"slugs": ["daimlertruck", "daimler", "dtna"], "sites": [
                "External", "Careers", "DaimlerTruckCareers", "Daimler_Careers",
                "daimler_jobs", "DTNACareers", "DTNA"
            ]}),
            ("smartrecruiters", "DaimlerTruckNorthAmerica"),
            ("smartrecruiters", "DaimlerTruck"),
            ("html", "https://careers.daimlertruck.com"),
            ("html", "https://jobs.daimlertruck.com"),
        ],
    },
    "Mercedes-Benz": {
        "strategies": [
            ("workday_discover", {"slugs": ["mercedesbenz", "mercedes", "mbrdna", "mbusa"], "sites": [
                "External", "Careers", "MBCareers", "MBRDNA", "MercedesBenz_Careers",
                "jobs", "Mercedes_Careers"
            ]}),
            ("smartrecruiters", "MercedesBenz"),
            ("smartrecruiters", "MercedesBenzResearchAndDevelopment"),
            ("greenhouse", "mbrdna"),
            ("html", "https://jobs.mercedes-benz.com"),
            ("html", "https://www.mbusa.com/en/careers"),
        ],
    },
    "GE Healthcare": {
        "strategies": [
            ("workday_discover", {"slugs": ["gehealthcare", "ge"], "sites": [
                "External", "Careers", "GEHealthCareCareers", "GE_HealthCare",
                "GEHealthCare", "GEHC_Careers", "GEHealthCare_External"
            ]}),
            ("html", "https://careers.gehealthcare.com"),
            ("html", "https://jobs.gecareers.com/healthcare"),
        ],
    },
}


def extract_skills(jobs):
    """Extract skill signals from job titles."""
    counts = {}
    for job in jobs:
        title_lower = job.get("title", "").lower()
        for kw, skill in SKILL_KEYWORDS.items():
            if kw in title_lower:
                counts[skill] = counts.get(skill, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: x[1], reverse=True))


def save_result(company, jobs, source_info):
    """Save scraped data to JSON."""
    safe_name = company.lower().replace(" ", "_").replace("/", "_").replace(",", "").replace(".", "")
    
    title_counts = {}
    loc_counts = {}
    for j in jobs:
        t = j.get("title", "")
        if t:
            title_counts[t] = title_counts.get(t, 0) + 1
        loc = j.get("location", "") or "Unknown"
        loc_counts[loc] = loc_counts.get(loc, 0) + 1
    
    output = {
        "company": company,
        "total_openings": len(jobs),
        "scraped_at": datetime.now().isoformat(),
        "source": source_info,
        "top_titles": dict(sorted(title_counts.items(), key=lambda x: x[1], reverse=True)[:25]),
        "top_locations": dict(sorted(loc_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
        "skill_signals": extract_skills(jobs),
        "jobs": jobs,
    }
    
    path = os.path.join(OUTPUT_DIR, f"{safe_name}.json")
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    return path


def scrape_company(company, config):
    """Try each strategy in order until one succeeds."""
    for strategy_type, params in config["strategies"]:
        
        if strategy_type == "workday_discover":
            slugs = params["slugs"]
            sites = params["sites"]
            print(f"    → Workday discovery ({len(slugs)} slugs × {len(sites)} sites)...", end=" ")
            for slug in slugs:
                url, total = discover_workday_url(slug, sites)
                if url:
                    print(f"✓ Found! ({total} total)")
                    jobs = scrape_workday(url)
                    if jobs:
                        return jobs, f"Workday API: {url}"
            print("✗")
        
        elif strategy_type == "greenhouse":
            token = params
            print(f"    → Greenhouse ({token})...", end=" ")
            jobs = scrape_greenhouse(token)
            if jobs:
                print(f"✓ {len(jobs)} jobs")
                return jobs, f"Greenhouse: {token}"
            print("✗")
        
        elif strategy_type == "lever":
            slug = params
            print(f"    → Lever ({slug})...", end=" ")
            jobs = scrape_lever(slug)
            if jobs:
                print(f"✓ {len(jobs)} jobs")
                return jobs, f"Lever: {slug}"
            print("✗")
        
        elif strategy_type == "smartrecruiters":
            company_id = params
            print(f"    → SmartRecruiters ({company_id})...", end=" ")
            jobs = scrape_smartrecruiters(company_id)
            if jobs:
                print(f"✓ {len(jobs)} jobs")
                return jobs, f"SmartRecruiters: {company_id}"
            print("✗")
        
        elif strategy_type == "html":
            url = params
            print(f"    → HTML ({url[:50]})...", end=" ")
            jobs = scrape_html_career_page(url)
            if jobs:
                print(f"✓ {len(jobs)} jobs")
                return jobs, f"HTML: {url}"
            print("✗")
        
        time.sleep(0.5)
    
    return [], "No working source found"


def main():
    print("=" * 70)
    print("  TARGETED CAREER PAGE SCRAPER — 10 PRIORITY COMPANIES")
    print("=" * 70)
    print(f"  Output: {OUTPUT_DIR}\n")
    
    results = {}
    
    for company, config in COMPANIES.items():
        print(f"\n  [{company}]")
        jobs, source = scrape_company(company, config)
        
        if jobs:
            path = save_result(company, jobs, source)
            results[company] = {"jobs": len(jobs), "source": source}
            skills = extract_skills(jobs)
            skills_str = ", ".join(list(skills.keys())[:5]) if skills else "none detected"
            print(f"    ✅ {len(jobs)} jobs | skills: {skills_str}")
            print(f"    Saved: {os.path.basename(path)}")
        else:
            results[company] = {"jobs": 0, "source": source}
            print(f"    ❌ No jobs found via any strategy")
    
    # Summary
    print(f"\n{'='*70}")
    print(f"  SCRAPING RESULTS")
    print(f"{'='*70}")
    
    succeeded = {c: r for c, r in results.items() if r["jobs"] > 0}
    failed = {c: r for c, r in results.items() if r["jobs"] == 0}
    
    total_jobs = sum(r["jobs"] for r in results.values())
    print(f"  Succeeded: {len(succeeded)}/{len(results)} companies")
    print(f"  Total new jobs: {total_jobs}")
    
    if succeeded:
        print(f"\n  ✅ SCRAPED:")
        for c, r in sorted(succeeded.items(), key=lambda x: x[1]["jobs"], reverse=True):
            print(f"    {c:<30} {r['jobs']:>5} jobs  ({r['source']})")
    
    if failed:
        print(f"\n  ❌ FAILED (may need manual career page URL):")
        for c in failed:
            print(f"    {c}")


if __name__ == "__main__":
    main()
