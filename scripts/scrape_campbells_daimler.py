"""Quick scrape of Campbells and Daimler with correct URLs."""
import requests
import json
import os
from datetime import datetime
from bs4 import BeautifulSoup

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'market_signals', 'career_pages')
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Accept": "text/html,application/json,*/*",
}

SKILL_KEYWORDS = {
    "python": "Python", "java ": "Java", "react": "React", "aws": "AWS",
    "azure": "Azure", "sql": "SQL", "ai": "AI", "cloud": "Cloud",
    "devops": "DevOps", "security": "Security", "sap": "SAP",
    "supply chain": "Supply Chain", "manufacturing": "Manufacturing",
    "quality": "Quality", "compliance": "Compliance", "data": "Data",
    "machine learning": "Machine Learning", "agile": "Agile",
}

def scrape_and_save(company, url):
    print(f"\n  [{company}] → {url}")
    jobs = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15, allow_redirects=True, verify=False)
        print(f"    Status: {resp.status_code}, Size: {len(resp.text)} bytes")
        soup = BeautifulSoup(resp.text, "html.parser")

        # Strategy 1: JSON-LD structured data
        for script in soup.select('script[type="application/ld+json"]'):
            try:
                ld = json.loads(script.string)
                items = ld if isinstance(ld, list) else [ld]
                for item in items:
                    if item.get("@type") == "JobPosting":
                        jobs.append({"title": item.get("title",""), "location": str(item.get("jobLocation","")), "posted_date": item.get("datePosted","")})
                    elif item.get("@type") == "ItemList":
                        for el in item.get("itemListElement", []):
                            if isinstance(el, dict) and el.get("@type") == "JobPosting":
                                jobs.append({"title": el.get("title",""), "location": str(el.get("jobLocation","")), "posted_date": el.get("datePosted","")})
            except:
                pass

        # Strategy 2: Links with job-related patterns
        seen = set()
        for link in soup.select('a[href*="job"], a[href*="career"], a[href*="position"], a[href*="opening"], a[href*="apply"], a.job-title, a.posting-title, h2 a, h3 a, div[class*="job"] a, li[class*="job"] a, a[class*="card"], a[class*="listing"]'):
            title = link.get_text(strip=True)
            if title and 5 < len(title) < 150 and title not in seen:
                seen.add(title)
                parent = link.find_parent(['li', 'div', 'tr', 'article'])
                location = ""
                if parent:
                    loc_el = parent.select_one('[class*="location"], [class*="Location"], span.location, [class*="city"]')
                    if loc_el:
                        location = loc_el.get_text(strip=True)
                jobs.append({"title": title, "location": location, "posted_date": ""})

        # Strategy 3: Any heading that looks like a job title
        if len(jobs) < 3:
            for tag in soup.select('h2, h3, h4'):
                text = tag.get_text(strip=True)
                if text and 8 < len(text) < 120 and text not in seen:
                    # Filter out navigation/section headers
                    lower = text.lower()
                    if any(w in lower for w in ['manager', 'engineer', 'analyst', 'developer', 'director',
                                                  'specialist', 'coordinator', 'lead', 'senior', 'associate',
                                                  'designer', 'scientist', 'technician', 'administrator']):
                        seen.add(text)
                        jobs.append({"title": text, "location": "", "posted_date": ""})

        print(f"    Found: {len(jobs)} jobs")
        if jobs:
            # Extract skills
            skills = {}
            titles = {}
            for j in jobs:
                t = j["title"]
                titles[t] = titles.get(t, 0) + 1
                for kw, skill in SKILL_KEYWORDS.items():
                    if kw in t.lower():
                        skills[skill] = skills.get(skill, 0) + 1

            safe = company.lower().replace(" ", "_").replace(",", "").replace(".", "")
            output = {
                "company": company, "total_openings": len(jobs),
                "scraped_at": datetime.now().isoformat(), "source": f"HTML: {url}",
                "top_titles": dict(sorted(titles.items(), key=lambda x: x[1], reverse=True)[:25]),
                "skill_signals": dict(sorted(skills.items(), key=lambda x: x[1], reverse=True)),
                "jobs": jobs,
            }
            path = os.path.join(OUTPUT_DIR, f"{safe}.json")
            with open(path, "w") as f:
                json.dump(output, f, indent=2)
            print(f"    ✅ Saved to {os.path.basename(path)}")
            for t, c in list(titles.items())[:10]:
                print(f"      • {t}")
        else:
            print(f"    ❌ No jobs extracted")
            # Dump page snippet for debug
            text = soup.get_text()[:500]
            print(f"    Page preview: {text[:200]}...")

    except Exception as e:
        print(f"    Error: {e}")

    return jobs

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

print("=" * 70)
print("  SCRAPE CAMPBELLS + DAIMLER (user-provided URLs)")
print("=" * 70)

r1 = scrape_and_save("Campbells", "https://careers.thecampbellscompany.com/")
r2 = scrape_and_save("Daimler", "https://www.daimlertruck.com/en/career")

print(f"\n  Total: Campbells={len(r1)}, Daimler={len(r2)}")
