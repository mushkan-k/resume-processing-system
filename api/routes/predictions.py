"""Predictions API — serves v4 job demand forecasts for the dashboard."""
import re
import json
from typing import Optional, List
from fastapi import APIRouter, Query
from api.database import db_pool

router = APIRouter(prefix="/api/predictions", tags=["Predictions"])

# ─── Location normalization (dedup city spelling variants) ───
CITY_NORMALIZE = {
    "Bangalor": "Bangalore",
    "Bengaluru": "Bangalore",
    "BENGALURU": "Bangalore",
    "BANGALORE": "Bangalore",
    "bengaluru": "Bangalore",
    "Hydrabad": "Hyderabad",
    "HYDERABAD": "Hyderabad",
    "Chenai": "Chennai",
    "CHENNAI": "Chennai",
    "PUNE": "Pune",
    "MUMBAI": "Mumbai",
    "Noida": "Noida",
    "NOIDA": "Noida",
    "Gurugram": "Gurgaon",
    "GURUGRAM": "Gurgaon",
}

# Map country codes to display regions
COUNTRY_TO_REGION = {
    "US": "US",
    "IN": "India",
    "AR": "LATAM", "BR": "LATAM", "CO": "LATAM", "MX": "LATAM", "CL": "LATAM",
    "CA": "Other", "DE": "Other", "FR": "Other", "IT": "Other", "PT": "Other",
    "AE": "Other", "AU": "Other",
}


def normalize_locations(locations: List[str]) -> List[str]:
    """Normalize and deduplicate location names."""
    if not locations:
        return []
    seen = set()
    result = []
    for loc in locations:
        loc = loc.strip()
        normalized = CITY_NORMALIZE.get(loc, loc)
        if normalized and normalized.lower() not in seen:
            seen.add(normalized.lower())
            result.append(normalized)
    return result


def clean_skills(skills: List[str]) -> List[str]:
    """Fix corrupted skills that were stored as 'or'-concatenated strings.
    
    e.g. "Compilersor Linkersor Debuggers" → ["Compilers", "Linkers", "Debuggers"]
         "Pythonor Numpyor Pandas" → ["Python", "Numpy", "Pandas"]
    """
    if not skills:
        return []
    cleaned = []
    for skill in skills:
        if not skill:
            continue
        # Detect concatenated "or" pattern: "Xor Y" or "X OR Y"
        # Pattern: lowercase "or" followed by space+uppercase, or " OR " with spaces
        if re.search(r'(?i)\bOR\b', skill) or re.search(r'[a-z]or\s+[A-Z]', skill):
            # Split by various OR patterns:
            # "X OR Y", "Xor Y" (no space before), "X or Y"
            parts = re.split(r'\s*[Oo][Rr]\s+|\s+[Oo][Rr]\s*', skill)
            for p in parts:
                p = p.strip('() ').strip()
                if p and len(p) > 1:
                    cleaned.append(p)
        else:
            cleaned.append(skill)
    
    # Deduplicate (case-insensitive) and filter junk
    seen = set()
    result = []
    SKIP_SKILLS = {'skills to be assigned', 'null', 'n/a', 'none'}
    for s in cleaned:
        if s.lower() not in seen and s.lower() not in SKIP_SKILLS and _is_clean_skill(s):
            seen.add(s.lower())
            result.append(s)
    return result


def get_cluster_region(cluster_name: str) -> str:
    """Extract region from cluster_name like 'IN | Software Engineer' → 'India'.
    For names without prefix, checks cache derived from actuals."""
    parts = cluster_name.split(" | ", 1)
    if len(parts) == 2:
        country = parts[0]
        return COUNTRY_TO_REGION.get(country, "Other")
    # Plain name — check if we cached the region from actuals
    if cluster_name in _cluster_region_cache:
        return _cluster_region_cache[cluster_name]
    return "All"


# Cache: maps plain cluster name → region derived from actuals
_cluster_region_cache: dict = {}


def get_cluster_region_from_db(cluster_name: str, cur) -> str:
    """Look up region for a plain cluster name by checking actuals with country prefix."""
    if cluster_name in _cluster_region_cache:
        return _cluster_region_cache[cluster_name]
    # Check if actuals exist with a country prefix for this role
    cur.execute("""
        SELECT cluster_name FROM demand_forecasts
        WHERE cluster_name LIKE %s AND data_type = 'actual'
        LIMIT 1
    """, (f"% | {cluster_name}",))
    row = cur.fetchone()
    if row:
        region = get_cluster_region(row['cluster_name'])
    else:
        region = "All"
    _cluster_region_cache[cluster_name] = region
    return region


def _quarter_to_month_range(quarter: str):
    """Convert '2026-Q1' → ('2026-01', '2026-03'). Returns None if invalid."""
    try:
        parts = quarter.split("-Q")
        if len(parts) != 2:
            return None
        year = parts[0]
        q = int(parts[1])
        month_ranges = {1: ("01", "03"), 2: ("04", "06"), 3: ("07", "09"), 4: ("10", "12")}
        if q not in month_ranges:
            return None
        start_m, end_m = month_ranges[q]
        return (f"{year}-{start_m}", f"{year}-{end_m}")
    except (ValueError, IndexError):
        return None


def _is_clean_skill(name: str) -> bool:
    """Filter out noisy/junk skill names from boolean JD fields."""
    if not name or len(name) < 2:
        return False
    if len(name) > 35:
        return False
    if '{' in name or '}' in name:
        return False
    if ' Or ' in name or ' or ' in name:
        return False
    if name.lower() in ('null', 'n/a', 'none', 'skills to be assigned'):
        return False
    return True


def _parse_json_field(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return None


def _normalize_skills_intelligence(raw_cats, market_context=None):
    """Support v2 flat categories and v3 historical/market/predicted structure."""
    if not raw_cats or not isinstance(raw_cats, dict):
        return None
    if "predicted" in raw_cats and "historical" in raw_cats:
        return raw_cats
    if "foundation" in raw_cats:
        return {
            "predicted": {
                "foundation": raw_cats.get("foundation", []),
                "trending_up": raw_cats.get("trending_up", []),
                "emerging": raw_cats.get("emerging", []),
                "stable": raw_cats.get("stable", []),
            },
            "historical": {
                "core": raw_cats.get("foundation", []),
                "recent": [],
                "momentum": raw_cats.get("trending_up", []),
                "stable": raw_cats.get("stable", []),
            },
            "market": {
                "hot": [{"skill": m["skill"], "score": m.get("score", 0), "category": m.get("category", "general")}
                        for m in (market_context or []) if m.get("trend") == "hot"][:6],
                "growing": [{"skill": m["skill"], "score": m.get("score", 0), "category": m.get("category", "general")}
                            for m in (market_context or []) if m.get("trend") in ("growing", "hot")][:6],
                "stable": [{"skill": m["skill"], "score": m.get("score", 0), "category": m.get("category", "general")}
                           for m in (market_context or []) if m.get("trend") == "stable"][:6],
            },
        }
    return None


def _skills_for_projected_role(hist_skills, skills_intelligence):
    """Blend historical role skills with categorized predicted skills for recruiters."""
    intel = skills_intelligence or {}
    predicted = intel.get("predicted", {})
    market = intel.get("market", {})
    foundation = predicted.get("foundation", [])
    trending = predicted.get("trending_up", [])
    emerging = predicted.get("emerging", [])
    hot_market = [m["skill"] if isinstance(m, dict) else m for m in market.get("hot", [])]

    pool = foundation + trending + emerging + hot_market
    pool_lower = {s.lower(): s for s in pool if s}

    chosen = []
    seen = set()
    for s in (hist_skills or []):
        key = s.lower()
        if key in pool_lower and key not in seen:
            chosen.append(pool_lower[key])
            seen.add(key)
    for group in (foundation, trending, emerging, hot_market):
        for s in group:
            key = s.lower()
            if key not in seen:
                chosen.append(s)
                seen.add(key)
            if len(chosen) >= 6:
                break
        if len(chosen) >= 6:
            break

    by_category = {
        "historical": [s for s in (hist_skills or [])[:4] if s],
        "predicted": [s for s in chosen if s.lower() in {x.lower() for x in foundation + trending + emerging}][:4],
        "market": [s for s in chosen if s.lower() in {x.lower() for x in hot_market}][:3],
    }
    return chosen[:6], by_category


def _get_extracted_skills(cur, cluster_name: str, company: str = None, quarter: str = None) -> tuple:
    """
    Get NLP-extracted skills from jd_extracted_skills table.
    Returns: (skills_list, jd_count, source_description)
    Priority: full-text NLP > boolean-parsed. Aggregates by frequency.
    If quarter is provided (e.g. "Q1 2026"), filter to that quarter's JDs.
    """
    conditions = []
    params = []

    role_part = cluster_name.split(" | ")[-1] if " | " in cluster_name else cluster_name
    country_part = cluster_name.split(" | ")[0] if " | " in cluster_name else ""

    if company:
        conditions.append("company_name LIKE %s")
        params.append(f"%{company}%")

    if country_part:
        conditions.append("country = %s")
        params.append(country_part)

    conditions.append("cluster_name LIKE %s")
    params.append(f"%{role_part}%")

    # Quarter filter: convert "Q1 2026" → "2026Q1"
    if quarter:
        q_db = None
        import re as _re
        m = _re.match(r'Q(\d)\s*(\d{4})', quarter)
        if m:
            q_db = f"{m.group(2)}Q{m.group(1)}"
        else:
            m2 = _re.match(r'(\d{4})Q(\d)', quarter)
            if m2:
                q_db = quarter
        if q_db:
            conditions.append("quarter = %s")
            params.append(q_db)

    where = " AND ".join(conditions)

    try:
        cur.execute(f"""
            SELECT extracted_skills, skill_count, source_type
            FROM jd_extracted_skills
            WHERE {where}
            ORDER BY
                CASE source_type WHEN 'jd_full_text' THEN 1 ELSE 2 END,
                skill_count DESC
            LIMIT 200
        """, params)
        rows = cur.fetchall()
    except Exception:
        return None, 0, ""

    if not rows:
        return None, 0, ""

    skill_freq = {}
    full_text_count = sum(1 for r in rows if r.get("source_type") == "jd_full_text")

    for row in rows:
        try:
            skills = json.loads(row["extracted_skills"])
        except Exception:
            continue
        for i, skill in enumerate(skills):
            weight = max(1, 10 - i)
            if row.get("source_type") == "jd_full_text":
                weight *= 3
            skill_freq[skill] = skill_freq.get(skill, 0) + weight

    SKIP = {"develop", "design", "web", "website", "work", "team", "support",
            "manage", "build", "create", "test", "project", "system", "data",
            "application", "software", "service", "experience", "years",
            "web or website", "null", "linkedin"}
    filtered = {k: v for k, v in skill_freq.items() if k.lower() not in SKIP and len(k) > 1}

    sorted_skills = sorted(filtered.keys(), key=lambda x: -filtered[x])[:8]

    source_desc = f"NLP-extracted from {len(rows)} JDs"
    if full_text_count > 0:
        source_desc += f" ({full_text_count} full-text)"

    return sorted_skills, len(rows), source_desc


@router.get("/companies")
def get_available_companies():
    """Get all companies that have cluster profiles — for the company filter dropdown."""
    conn = db_pool.get_connection()
    try:
        cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT DISTINCT company_name
            FROM company_cluster_profiles
            WHERE company_name IS NOT NULL AND company_name != ''
            ORDER BY company_name
        """)
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()
    return [r["company_name"] for r in rows]


@router.get("")
def get_predictions(
    quarter: Optional[str] = Query(None, description="e.g. 2026-Q3"),
    reliable_only: bool = Query(True, description="Only show Grade A/B/C clusters"),
    region: Optional[str] = Query(None, description="Filter by region: India, US, LATAM, Other"),
    company: Optional[str] = Query(None, description="Filter by company name (partial match)"),
):
    """Dashboard overview: quarterly demand forecasts by cluster.

    Aggregates monthly predictions into quarters for display.
    """
    conn = db_pool.get_connection()
    try:
        cur = conn.cursor(dictionary=True)

        sql = "SELECT * FROM demand_forecasts"
        if reliable_only:
            sql += " WHERE is_reliable = TRUE"
        sql += " ORDER BY cluster_name, forecast_month"

        cur.execute(sql)
        rows = cur.fetchall()

        # Load company-specific profiles if company filter is set
        company_clusters = None
        company_share = {}  # cluster_name -> proportion (0.0 to 1.0)
        if company:
            cur.execute("""
                SELECT c.cluster_name, c.company_name, c.jd_count, c.openings as company_openings,
                       t.total_jds, t.total_openings
                FROM company_cluster_profiles c
                JOIN (
                    SELECT cluster_name, SUM(jd_count) as total_jds, SUM(openings) as total_openings
                    FROM company_cluster_profiles
                    GROUP BY cluster_name
                ) t ON c.cluster_name = t.cluster_name
                WHERE c.company_name LIKE %s
            """, (f"%{company}%",))
            company_rows = cur.fetchall()
            company_clusters = set()
            company_names = {}
            for cr in company_rows:
                cn = cr["cluster_name"]
                company_clusters.add(cn)
                company_names[cn] = cr["company_name"]
                # Proportional share based on openings (more accurate than JD count)
                if cr["total_openings"] and cr["total_openings"] > 0:
                    company_share[cn] = cr["company_openings"] / cr["total_openings"]
                elif cr["total_jds"] and cr["total_jds"] > 0:
                    company_share[cn] = cr["jd_count"] / cr["total_jds"]
                else:
                    company_share[cn] = 1.0

        cur.close()
    finally:
        conn.close()

    if not rows:
        return {"quarters": [], "clusters": [], "summary": {}}

    # Group by cluster and quarter
    clusters_data = {}
    for row in rows:
        cluster = row["cluster_name"]

        # ─── Region filter: skip clusters not matching ───
        # Same fix as the /clusters endpoint: frontend now sends a combined
        # "LATAM & Others" value for anything non-US/non-India, so match on
        # that basis instead of exact string equality.
        if region:
            _region_lower = region.lower()
            _this_cluster_region = get_cluster_region(cluster)
            if _region_lower == "us":
                if _this_cluster_region != "US":
                    continue
            elif _region_lower == "india":
                if _this_cluster_region != "India":
                    continue
            else:
                if _this_cluster_region in ("US", "India"):
                    continue

        # ─── Company filter: only show clusters this company is in ───
        if company and company_clusters and cluster not in company_clusters:
            continue

        month = row["forecast_month"]  # e.g. "2026-07"

        # Determine quarter
        year, mon = month.split("-")
        mon_int = int(mon)
        if mon_int <= 3:
            q = f"{year}-Q1"
        elif mon_int <= 6:
            q = f"{year}-Q2"
        elif mon_int <= 9:
            q = f"{year}-Q3"
        else:
            q = f"{year}-Q4"

        # Initialize cluster on first encounter (before quarter filter)
        if cluster not in clusters_data:
            clusters_data[cluster] = {
                "cluster": cluster,
                "model": row["model_used"],
                "mape": float(row["mape"]) if row["mape"] else None,
                "mae": float(row["mae"]) if row["mae"] else None,
                "mase": float(row["mase"]) if row["mase"] else None,
                "topSkills": clean_skills(json.loads(row["top_skills"]) if row["top_skills"] else []),
                "topLocations": normalize_locations(json.loads(row["top_locations"]) if row.get("top_locations") else []),
                "topClients": [company_names[cluster]] if company and cluster in company_names else (json.loads(row["top_clients"]) if row.get("top_clients") else []),
                "reliable": bool(row["is_reliable"]),
                "hasForecasts": False,
                "quarters": {},
            }

        # Track if this cluster has any predicted data (even outside current quarter filter)
        data_type = row.get("data_type", "predicted")
        if data_type != "actual":
            clusters_data[cluster]["hasForecasts"] = True

        if quarter and q != quarter:
            continue

        if q not in clusters_data[cluster]["quarters"]:
            clusters_data[cluster]["quarters"][q] = {
                "quarter": q,
                "totalDemand": 0,
                "lowerBound": 0,
                "upperBound": 0,
                "dataType": data_type,  # "actual" or "predicted"
                "months": [],
            }

        # Apply proportional share if company filter is active
        share = company_share.get(cluster, 1.0) if company else 1.0
        demand = max(1, round(row["demand_predicted"] * share)) if share < 1.0 else row["demand_predicted"]
        lower = max(0, round(row["demand_lower"] * share)) if share < 1.0 else row["demand_lower"]
        upper = max(1, round(row["demand_upper"] * share)) if share < 1.0 else row["demand_upper"]

        clusters_data[cluster]["quarters"][q]["totalDemand"] += demand
        clusters_data[cluster]["quarters"][q]["lowerBound"] += lower
        clusters_data[cluster]["quarters"][q]["upperBound"] += upper
        clusters_data[cluster]["quarters"][q]["months"].append({
            "month": month,
            "demand": demand,
            "lower": lower,
            "upper": upper,
            "dataType": data_type,
        })

        # If quarter has mixed types, mark as "mixed"
        if clusters_data[cluster]["quarters"][q]["dataType"] != data_type:
            clusters_data[cluster]["quarters"][q]["dataType"] = "mixed"

    # Compute grade per cluster (only meaningful when we have predictions)
    for cluster_info in clusters_data.values():
        mape = cluster_info["mape"]
        mase = cluster_info["mase"]
        has_forecasts = cluster_info.get("hasForecasts", False)

        if not has_forecasts:
            # Actuals-only cluster — no grade/accuracy applicable
            cluster_info["grade"] = None
            cluster_info["accuracy"] = None
            cluster_info["confidence"] = "actual"
        elif mape is not None:
            if mape <= 20:
                cluster_info["grade"] = "A"
                cluster_info["confidence"] = "high"
            elif mape <= 35:
                cluster_info["grade"] = "B"
                cluster_info["confidence"] = "high"
            elif mape <= 50:
                cluster_info["grade"] = "C"
                cluster_info["confidence"] = "medium"
            else:
                cluster_info["grade"] = "D"
                cluster_info["confidence"] = "low"
            cluster_info["accuracy"] = round(100 - mape, 1)
        else:
            cluster_info["grade"] = "D"
            cluster_info["accuracy"] = None
            cluster_info["confidence"] = "low"

    all_quarters = sorted(set(
        q for c in clusters_data.values() for q in c["quarters"]
    ))

    total_demand = sum(
        sum(q["totalDemand"] for q in c["quarters"].values())
        for c in clusters_data.values()
    )

    return {
        "quarters": all_quarters,
        "totalDemand6Months": total_demand,
        "clustersCount": len(clusters_data),
        "clusters": sorted(
            clusters_data.values(),
            key=lambda x: sum(q["totalDemand"] for q in x["quarters"].values()),
            reverse=True,
        ),
    }


@router.get("/clusters")
def get_cluster_list(
    region: Optional[str] = Query(None, description="Filter by region: India, US, LATAM, Other"),
    company: Optional[str] = Query(None, description="Filter by company name (partial match)"),
    quarter: Optional[str] = Query(None, description="Only show clusters with data in this quarter, e.g. 2026-Q3"),
    division: Optional[str] = Query(None, description="Filter by division: Talent-on-Demand, Services"),
):
    """All clusters with accuracy grades. Pass ?region=India&company=Wabtec&quarter=2026-Q3&division=Talent-on-Demand for filtered results."""
    conn = db_pool.get_connection()
    try:
        cur = conn.cursor(dictionary=True)

        # If quarter filter is set, only get clusters that have data in that quarter
        quarter_filter_sql = ""
        if quarter:
            # Convert quarter to month range (e.g. "2026-Q3" → "2026-07" to "2026-09")
            parts = quarter.split("-Q")
            if len(parts) == 2:
                year = parts[0]
                q = int(parts[1])
                month_ranges = {1: ("01","03"), 2: ("04","06"), 3: ("07","09"), 4: ("10","12")}
                if q in month_ranges:
                    start_m, end_m = month_ranges[q]
                    quarter_filter_sql = f" AND forecast_month BETWEEN '{year}-{start_m}' AND '{year}-{end_m}'"

        cur.execute(f"""
            SELECT cluster_name, model_used, mape, mae, mase,
                   is_reliable, top_skills, top_locations, top_clients,
                   SUM(demand_predicted) as total_demand,
                   MIN(forecast_month) as first_month,
                   MAX(forecast_month) as last_month,
                   MAX(CASE WHEN data_type = 'predicted' THEN 1 ELSE 0 END) as has_forecasts
            FROM demand_forecasts
            WHERE 1=1 {quarter_filter_sql}
            GROUP BY cluster_name, model_used, mape, mae, mase, is_reliable, top_skills, top_locations, top_clients
            HAVING SUM(demand_predicted) > 0
            ORDER BY total_demand DESC
        """)
        clusters = cur.fetchall()

        # Load company-specific profiles if company filter is set
        company_profiles = {}
        company_profiles_by_role = {}  # plain role name → profile (for matching predicted clusters)
        company_share_summary = {}  # cluster_name -> this company's proportional share of demand
        company_share_summary_by_role = {}  # plain role fallback for predicted clusters (no region prefix)
        if company:
            cur.execute("""
                SELECT cluster_name, company_name, top_skills, top_locations, jd_count
                FROM company_cluster_profiles
                WHERE company_name LIKE %s
                ORDER BY cluster_name
            """, (f"%{company}%",))
            for row in cur.fetchall():
                company_profiles[row["cluster_name"]] = row
                # Also index by plain role name for predicted clusters without prefix
                parts = row["cluster_name"].split(" | ", 1)
                plain_role = parts[1] if len(parts) == 2 else row["cluster_name"]
                company_profiles_by_role[plain_role] = row

            # Compute this company's proportional share of demand per cluster
            # (same logic as /summary — otherwise totalDemand below shows the
            # FULL cluster total instead of this company's slice of it)
            cur.execute("""
                SELECT c.cluster_name, c.openings as company_openings,
                       t.total_openings, c.jd_count, t.total_jds
                FROM company_cluster_profiles c
                JOIN (
                    SELECT cluster_name, SUM(jd_count) as total_jds, SUM(openings) as total_openings
                    FROM company_cluster_profiles
                    GROUP BY cluster_name
                ) t ON c.cluster_name = t.cluster_name
                WHERE c.company_name LIKE %s
            """, (f"%{company}%",))
            for cr in cur.fetchall():
                cn = cr["cluster_name"]
                if cr["total_openings"] and cr["total_openings"] > 0:
                    share_val = cr["company_openings"] / cr["total_openings"]
                elif cr["total_jds"] and cr["total_jds"] > 0:
                    share_val = cr["jd_count"] / cr["total_jds"]
                else:
                    share_val = 1.0
                company_share_summary[cn] = share_val
                # Also index by plain role for predicted clusters (no region prefix)
                _parts_share = cn.split(" | ", 1)
                _plain_share = _parts_share[1] if len(_parts_share) == 2 else cn
                company_share_summary_by_role[_plain_share] = share_val

        # Pre-load region mapping from actuals for clusters without country prefix
        # A plain cluster name may map to multiple regions (e.g. "Software Engineer" → India, US)
        # We pick the region with the most demand as the "primary" region
        cur.execute("""
            SELECT SUBSTRING_INDEX(cluster_name, ' | ', 1) as country_code,
                   SUBSTRING_INDEX(cluster_name, ' | ', -1) as plain_name,
                   SUM(demand_predicted) as total
            FROM demand_forecasts
            WHERE data_type = 'actual' AND cluster_name LIKE '%% | %%'
            GROUP BY country_code, plain_name
            ORDER BY plain_name, total DESC
        """)
        _region_scores: dict = {}  # plain_name → {region: total_demand}
        for row in cur.fetchall():
            plain = row['plain_name']
            region_val = COUNTRY_TO_REGION.get(row['country_code'], "Other")
            if plain not in _region_scores:
                _region_scores[plain] = {}
            _region_scores[plain][region_val] = _region_scores[plain].get(region_val, 0) + int(row['total'])
        
        for plain, scores in _region_scores.items():
            # Primary region = highest demand
            primary = max(scores, key=scores.get)
            _cluster_region_cache[plain] = primary

        # Load division map if division filter is set
        division_map = {}
        if division:
            cur2 = conn.cursor(dictionary=True)
            cur2.execute("SELECT cluster_name, division FROM cluster_division_map")
            for row in cur2.fetchall():
                division_map[row["cluster_name"].lower()] = row["division"]
            cur2.close()

        cur.close()
    finally:
        conn.close()

    result = []
    for c in clusters:
        cluster_name = c["cluster_name"]

        # ─── Region filter: skip clusters not matching requested region ───
        # The frontend now offers a single combined "LATAM & Others" option
        # instead of separate LATAM/Other choices, and sends that exact string
        # as the region param. Rather than hardcode that literal string (fragile
        # if the label ever changes), treat any region value that isn't exactly
        # "US" or "India" as "everything else" — which is what the UI now means
        # by any non-US/non-India selection.
        cluster_region = get_cluster_region(cluster_name)
        if region:
            _region_lower = region.lower()
            if _region_lower == "us":
                if cluster_region != "US":
                    continue
            elif _region_lower == "india":
                if cluster_region != "India":
                    continue
            else:
                # Any other region selection (LATAM, Other, "LATAM & Others", etc.)
                # means "not US and not India"
                if cluster_region in ("US", "India"):
                    continue

        # ─── Division filter: skip clusters not in requested division ───
        if division:
            # Get plain role name from cluster (strip "US | " prefix)
            _parts_div = cluster_name.split(" | ", 1)
            _plain_div = _parts_div[1].lower() if len(_parts_div) == 2 else cluster_name.lower()
            cluster_div = division_map.get(_plain_div, division_map.get(cluster_name.lower()))
            if not cluster_div:
                continue  # cluster not in division map at all
            if cluster_div != "Both" and cluster_div.lower().replace("-", " ") != division.lower().replace("-", " "):
                continue

        # ─── Company filter: only show clusters this company is in ───
        if company:
            # Match by full name OR plain role name
            _parts_co = cluster_name.split(" | ", 1)
            _plain_co = _parts_co[1] if len(_parts_co) == 2 else cluster_name
            if cluster_name not in company_profiles and _plain_co not in company_profiles_by_role:
                continue
            # Use the matched profile for skills/locations below
            if cluster_name in company_profiles:
                company_profiles[cluster_name] = company_profiles[cluster_name]
            elif _plain_co in company_profiles_by_role:
                company_profiles[cluster_name] = company_profiles_by_role[_plain_co]

        mape = float(c["mape"]) if c["mape"] else None
        mase = float(c["mase"]) if c["mase"] else None
        has_forecasts = bool(c.get("has_forecasts", 1))

        if not has_forecasts:
            grade = None
        elif mape is not None:
            if mape <= 35:
                grade = "Very Stable"
            elif mape <= 50:
                grade = "Stable"
            elif mape <= 70:
                grade = "Moderate"
            else:
                grade = "Volatile"
        else:
            grade = "Volatile"

        # Use company-specific skills/locations if available
        skills_source = "all"
        if company and cluster_name in company_profiles:
            cp = company_profiles[cluster_name]
            skills = json.loads(cp["top_skills"]) if cp["top_skills"] else []
            locations = json.loads(cp["top_locations"]) if cp["top_locations"] else []
            skills_source = company
        else:
            skills = json.loads(c["top_skills"]) if c["top_skills"] else []
            locations = json.loads(c["top_locations"]) if c.get("top_locations") else []

        # Parse cluster into region/role parts
        parts = cluster_name.split(" | ", 1)
        country = parts[0] if len(parts) == 2 else ""
        role = parts[1] if len(parts) == 2 else cluster_name

        if company and cluster_name in company_profiles:
            raw_clients = [company_profiles[cluster_name].get("company_name", company)]
        else:
            raw_clients = json.loads(c["top_clients"]) if c.get("top_clients") else []

        # For actuals-only clusters, compute a trend estimate for display
        total_demand_val = int(c["total_demand"])

        # ─── Company filter: scale demand down to this company's proportional share ───
        # Without this, totalDemand still shows the FULL cluster total even when
        # a company filter is active — it only affected which clusters were shown.
        # Role-only fallback exists ONLY for predicted clusters that have no region
        # prefix at all (e.g. "Specialist - General", not "US | Specialist - General").
        # It must NOT fire when this row DOES have a specific region prefix (e.g.
        # "DE | Specialist - General") but the company simply has zero data there —
        # that's a real "no footprint in this region" case, not a missing-prefix case,
        # and borrowing another region's share would fabricate a number for a region
        # the company has never actually posted in.
        if company:
            share = company_share_summary.get(cluster_name)
            if share is None and " | " not in cluster_name:
                share = company_share_summary_by_role.get(role)  # only for genuinely region-less predicted names
            if share is not None:
                total_demand_val = max(1, round(total_demand_val * share))
            elif " | " in cluster_name:
                # Region-specific cluster, company has zero recorded presence here —
                # honest answer is zero, not the unscaled full-cluster total.
                total_demand_val = 0

        if not has_forecasts and total_demand_val > 0:
            # Monthly avg from actuals → project 6 months
            # total_demand is sum of actuals (Jan-May = 5 months)
            monthly_avg = total_demand_val / 5.0
            estimated_6m = round(monthly_avg * 6)
            confidence_label = "estimate"
        else:
            estimated_6m = 0
            confidence_label = "high" if grade in ("Very Stable", "Stable") else ("medium" if grade == "Moderate" else "low")

        # When a company filter is active and this cluster genuinely has zero
        # demand for that company (confirmed zero, not an unknown/unscaled value),
        # skip it rather than showing an empty "0 jobs" card — a recruiter
        # filtering by company wants to see where that company DOES hire, not
        # a wall of confirmed-empty clusters.
        if company and total_demand_val == 0:
            continue

        result.append({
            "cluster": cluster_name,
            "country": country,
            "region": cluster_region,
            "role": role,
            "model": c["model_used"],
            "totalDemand": total_demand_val,
            "estimatedDemand": estimated_6m,
            "forecastRange": f"{c['first_month']} to {c['last_month']}",
            "mape": mape,
            "accuracy": round(100 - mape, 1) if mape else None,
            "mase": mase,
            "grade": grade,
            "reliable": True if not has_forecasts else bool(c["is_reliable"]),
            "hasForecasts": has_forecasts,
            "confidence": confidence_label,
            "topSkills": clean_skills(skills),
            "topLocations": normalize_locations(locations),
            "topClients": raw_clients,
            "skillsSource": skills_source,
        })

    # ─── Deduplicate: merge actual + predicted clusters for same country+role ───
    # e.g. "Java Developer" (predicted, no prefix) + "IN | Java Developer" (actual) → one entry
    deduped = {}
    for c in result:
        key = (c["country"], c["role"])
        if key not in deduped:
            deduped[key] = c
        else:
            existing = deduped[key]
            # Prefer the one with forecasts; merge demand numbers
            if c["hasForecasts"] and not existing["hasForecasts"]:
                # Predicted cluster wins — add actual demand from the other
                c["actualDemand"] = existing.get("totalDemand", 0)
                c["totalDemand"] = c["totalDemand"] + existing.get("totalDemand", 0)
                # Merge skills/locations if the predicted one is empty
                if not c["topSkills"] and existing["topSkills"]:
                    c["topSkills"] = existing["topSkills"]
                if not c["topLocations"] and existing["topLocations"]:
                    c["topLocations"] = existing["topLocations"]
                if not c["topClients"] and existing["topClients"]:
                    c["topClients"] = existing["topClients"]
                deduped[key] = c
            elif existing["hasForecasts"] and not c["hasForecasts"]:
                # Existing predicted wins — add actual demand from new
                existing["actualDemand"] = c.get("totalDemand", 0)
                existing["totalDemand"] = existing["totalDemand"] + c.get("totalDemand", 0)
                if not existing["topSkills"] and c["topSkills"]:
                    existing["topSkills"] = c["topSkills"]
                if not existing["topLocations"] and c["topLocations"]:
                    existing["topLocations"] = c["topLocations"]
                if not existing["topClients"] and c["topClients"]:
                    existing["topClients"] = c["topClients"]
            else:
                # Both same type — keep the one with higher demand
                if c["totalDemand"] > existing["totalDemand"]:
                    deduped[key] = c

    result = sorted(deduped.values(), key=lambda x: x["totalDemand"], reverse=True)

    return {
        "clusters": result,
        "total": len(result),
        "filters": {"region": region, "company": company},
    }


@router.get("/cluster/{cluster_name:path}")
def get_cluster_detail(
    cluster_name: str,
    company: Optional[str] = Query(None, description="Show company-specific skills"),
    quarter: Optional[str] = Query(None, description="Quarter for skills e.g. Q1 2026"),
):
    """Monthly predictions for a specific cluster. Pass ?company=X for company-specific skills, ?quarter=Q1 2026 for quarter-specific skills."""
    conn = db_pool.get_connection()
    try:
        cur = conn.cursor(dictionary=True)

        # If quarter is specified, filter to that quarter's months
        quarter_filter_sql = ""
        quarter_params = [cluster_name]
        if quarter:
            # quarter format: "Q1 2026" → convert to month range
            parts = quarter.split(" ")
            if len(parts) == 2:
                q_num = int(parts[0][1])  # "Q1" → 1
                year = parts[1]
                month_ranges = {1: ("01","03"), 2: ("04","06"), 3: ("07","09"), 4: ("10","12")}
                if q_num in month_ranges:
                    start_m, end_m = month_ranges[q_num]
                    quarter_filter_sql = " AND forecast_month BETWEEN %s AND %s"
                    quarter_params.extend([f"{year}-{start_m}", f"{year}-{end_m}"])

        cur.execute(f"""
            SELECT * FROM demand_forecasts
            WHERE cluster_name = %s{quarter_filter_sql}
            ORDER BY forecast_month
        """, quarter_params)
        rows = cur.fetchall()

        if not rows:
            cur.close()
            return {"error": f"Cluster '{cluster_name}' not found"}

        # ─── Company filter: compute this company's real proportional share ───
        # ONE computation, used consistently for every number returned below
        # (per-month rows AND the aggregate KPIs) — this replaces two separate,
        # inconsistent scaling attempts that used to exist: one that silently
        # did nothing when its lookup missed (showing full unscaled totals
        # mislabeled as "Scoped to X"), and one that only ever scaled the
        # aggregate totals, never the per-month table underneath them.
        _co_share = None  # None = no company filter, or no data found for this company+cluster
        _co_share_found = False
        if company:
            _parts_cs = cluster_name.split(" | ", 1)
            _plain_cs = _parts_cs[1] if len(_parts_cs) == 2 else cluster_name
            _has_region_prefix = len(_parts_cs) == 2
            if _has_region_prefix:
                # This cluster already names a specific region (e.g. "DE | Specialist - General").
                # Match ONLY that exact cluster — do not fall back to a LIKE match, which would
                # silently pull a DIFFERENT region's data (e.g. "US | Specialist - General") just
                # because it shares the same role name. A company with zero presence in this
                # specific region must show zero, not another region's number borrowed in.
                cur.execute("""
                    SELECT c.openings as company_openings, t.total_openings, c.jd_count, t.total_jds
                    FROM company_cluster_profiles c
                    JOIN (
                        SELECT cluster_name, SUM(jd_count) as total_jds, SUM(openings) as total_openings
                        FROM company_cluster_profiles GROUP BY cluster_name
                    ) t ON c.cluster_name = t.cluster_name
                    WHERE c.company_name LIKE %s AND c.cluster_name = %s
                    LIMIT 1
                """, (f"%{company}%", cluster_name))
            else:
                # Genuinely region-less predicted cluster name — the LIKE fallback here is
                # legitimate, since there's no region info in this name to contradict.
                cur.execute("""
                    SELECT c.openings as company_openings, t.total_openings, c.jd_count, t.total_jds
                    FROM company_cluster_profiles c
                    JOIN (
                        SELECT cluster_name, SUM(jd_count) as total_jds, SUM(openings) as total_openings
                        FROM company_cluster_profiles GROUP BY cluster_name
                    ) t ON c.cluster_name = t.cluster_name
                    WHERE c.company_name LIKE %s AND c.cluster_name LIKE %s
                    LIMIT 1
                """, (f"%{company}%", f"% | {_plain_cs}"))
            _share_row = cur.fetchone()
            if _share_row:
                if _share_row["total_openings"] and _share_row["total_openings"] > 0:
                    _co_share = _share_row["company_openings"] / _share_row["total_openings"]
                elif _share_row["total_jds"] and _share_row["total_jds"] > 0:
                    _co_share = _share_row["jd_count"] / _share_row["total_jds"]
                else:
                    _co_share = 1.0
                _co_share_found = True
            elif _has_region_prefix:
                # Region-specific cluster, no company data found for THIS region —
                # honest answer is the company has zero recorded presence here.
                _co_share = 0.0
                _co_share_found = True
            # If no row found AND no region prefix: _co_share stays None. We do NOT
            # silently fall back to showing the full unscaled cluster — that's what
            # caused the "Scoped to T-Mobile" label on numbers that weren't actually
            # T-Mobile's. Below, scope_note makes this explicit instead.


        # Load company-specific profile if requested
        company_profile = None
        if company:
            cur.execute("""
                SELECT top_skills, top_locations, jd_count
                FROM company_cluster_profiles
                WHERE company_name LIKE %s AND cluster_name = %s
                LIMIT 1
            """, (f"%{company}%", cluster_name))
            company_profile = cur.fetchone()

        first_row = rows[0]
        # Get metadata from a predicted row (actuals may have NULL for mape/skills)
        predicted_rows = [r for r in rows if r.get("data_type") != "actual"]
        actual_rows = [r for r in rows if r.get("data_type") == "actual"]
        has_predictions = len(predicted_rows) > 0
        meta_row = predicted_rows[0] if predicted_rows else first_row

        # For actuals-only clusters, mape/mase are meaningless
        if has_predictions:
            mape = float(meta_row["mape"]) if meta_row["mape"] else None
            mase = float(meta_row["mase"]) if meta_row["mase"] else None
        else:
            # Actuals-only: no model was used, no accuracy to report
            mape = None
            mase = None

        # Compute grade — only meaningful when we have predictions
        if not has_predictions:
            grade = None  # actuals-only, no grade
        elif mape is not None:
            if mape <= 35:
                grade = "Very Stable"
            elif mape <= 50:
                grade = "Stable"
            elif mape <= 70:
                grade = "Moderate"
            else:
                grade = "Volatile"
        else:
            grade = "Volatile"

        predictions = []
        actual_demand = 0
        predicted_demand = 0
        _share_for_scaling = _co_share if _co_share is not None else 1.0

        for row in rows:
            demand = row["demand_predicted"]
            lower = row["demand_lower"]
            upper = row["demand_upper"]
            data_type = row.get("data_type", "predicted")

            # Scale to company share (no-op when no company filter, since
            # _share_for_scaling is 1.0 in that case)
            if _share_for_scaling == 0:
                # True zero — company has no recorded presence in this specific
                # region/cluster. Must show 0, not get floored up to 1 by the
                # min-1 safety net below (that net exists for fractional shares
                # like 0.004, not for a genuine zero).
                demand = 0
                lower = 0
                upper = 0
            elif _share_for_scaling != 1.0:
                demand = max(1, round(demand * _share_for_scaling)) if demand else 0
                lower = max(0, round((lower or 0) * _share_for_scaling))
                upper = max(demand, round((upper or 0) * _share_for_scaling))

            if data_type == "actual":
                # ACTUAL: real recorded data — no range, no confidence needed
                actual_demand += demand
                predictions.append({
                    "month": row["forecast_month"],
                    "type": "actual",
                    "demand": demand,
                    "planFor": f"Recorded: {demand} hires",
                })
            else:
                # FORECAST: model prediction — show range + confidence
                predicted_demand += demand
                spread = upper - lower
                if demand > 0:
                    spread_ratio = spread / demand
                    if spread_ratio <= 1.0:
                        confidence = "high"
                    elif spread_ratio <= 2.0:
                        confidence = "medium"
                    else:
                        confidence = "low"
                else:
                    confidence = "low"

                predictions.append({
                    "month": row["forecast_month"],
                    "type": "predicted",
                    "demand": demand,
                    "lower": lower,
                    "upper": upper,
                    "confidence": confidence,
                    "planFor": f"Expect ~{demand} hires (range {lower}\u2013{upper})",
                })

        # ─── For actuals-only clusters: generate trend estimates for future months ───
        estimated_demand = 0
        if not has_predictions and actual_demand > 0 and not quarter:
            # Calculate monthly run rate from actuals
            actual_months = [r for r in rows if r.get("data_type") == "actual"]
            num_months = len(actual_months)
            monthly_avg = actual_demand / max(num_months, 1)

            # Project forward: Jun (rest of Q2) + Q3 (Jul-Sep) + Q4 (Oct-Dec)
            future_months = ["2026-06", "2026-07", "2026-08", "2026-09", "2026-10", "2026-11", "2026-12"]
            existing_months = set(r["forecast_month"] for r in rows)

            for fm in future_months:
                if fm in existing_months:
                    continue
                # Simple estimate: monthly average with ±50% range (low confidence)
                est = max(1, round(monthly_avg))
                lower_est = max(0, round(monthly_avg * 0.5))
                upper_est = max(1, round(monthly_avg * 1.8))
                estimated_demand += est
                predictions.append({
                    "month": fm,
                    "type": "estimate",
                    "demand": est,
                    "lower": lower_est,
                    "upper": upper_est,
                    "confidence": "low",
                    "planFor": f"Trend estimate: ~{est}/month (based on {actual_demand} hires in {num_months} months)",
                })

        # Overall cluster confidence — derived from grade
        total_demand = actual_demand + predicted_demand + estimated_demand
        if not has_predictions and estimated_demand > 0:
            cluster_confidence = "estimate"
        elif not has_predictions:
            cluster_confidence = "actual"
        elif grade in ("Very Stable", "Stable"):
            cluster_confidence = "high"
        elif grade == "Moderate":
            cluster_confidence = "medium"
        else:
            cluster_confidence = "low"

        # Use NLP-extracted skills as PRIMARY source (from actual JD text)
        nlp_skills, nlp_count, nlp_source = _get_extracted_skills(cur, cluster_name, company=company, quarter=quarter)

        # For predicted quarters with poor/no skills, fall back to all-time extraction
        if (not nlp_skills or nlp_count < 5) and quarter:
            nlp_skills_all, nlp_count_all, nlp_source_all = _get_extracted_skills(cur, cluster_name, company=company, quarter=None)
            if nlp_skills_all and nlp_count_all > nlp_count:
                nlp_skills, nlp_count, nlp_source = nlp_skills_all, nlp_count_all, nlp_source_all

        if nlp_skills:
            skills = nlp_skills
            _nlp_skills_note = nlp_source
        elif company and company_profile:
            skills = json.loads(company_profile["top_skills"]) if company_profile.get("top_skills") else []
            _nlp_skills_note = f"From {company} company profile"
        else:
            skills = json.loads(meta_row["top_skills"]) if meta_row.get("top_skills") else []
            _nlp_skills_note = "From forecast model"

        skills_intelligence = None
        # For actual quarters, build skillsIntelligence from skill_trends + marketContext
        # (gets populated later after we compute skill_evolution and market_context)

        # Locations
        if company and company_profile:
            locations = json.loads(company_profile["top_locations"]) if company_profile.get("top_locations") else []
        else:
            locations = json.loads(meta_row["top_locations"]) if meta_row.get("top_locations") else []

        # Clients
        raw_clients = json.loads(meta_row["top_clients"]) if meta_row.get("top_clients") else []

        # Fallback: if predicted rows have no locations/clients, pull from actuals
        if not locations or not raw_clients:
            _parts_fb = cluster_name.split(" | ", 1)
            _plain_fb = _parts_fb[1] if len(_parts_fb) == 2 else cluster_name
            cur.execute("""
                SELECT top_locations, top_clients FROM demand_forecasts
                WHERE cluster_name LIKE %s AND data_type = 'actual'
                  AND (top_locations IS NOT NULL OR top_clients IS NOT NULL)
                ORDER BY forecast_month DESC LIMIT 1
            """, (f"% | {_plain_fb}",))
            fb_row = cur.fetchone()
            if fb_row:
                if not locations and fb_row.get("top_locations"):
                    locations = json.loads(fb_row["top_locations"])
                if not raw_clients and fb_row.get("top_clients"):
                    raw_clients = json.loads(fb_row["top_clients"])

        # Fallback: if still no clients, pull from company_cluster_profiles
        if not raw_clients:
            _parts_cp = cluster_name.split(" | ", 1)
            _plain_cp = _parts_cp[1] if len(_parts_cp) == 2 else cluster_name
            cur.execute("""
                SELECT company_name, openings FROM company_cluster_profiles
                WHERE cluster_name = %s OR cluster_name LIKE %s
                ORDER BY openings DESC LIMIT 10
            """, (cluster_name, f"% | {_plain_cp}"))
            cp_rows = cur.fetchall()
            if cp_rows:
                raw_clients = [r["company_name"] for r in cp_rows]

        # ─── Company filter: show only the filtered company in the clients list ───
        if company:
            raw_clients = [rc for rc in raw_clients if company.lower() in rc.lower()]
            if not raw_clients:
                raw_clients = [company]

        # ─── Hiring History (quarterly) ───
        # Parse plain role name from cluster_name
        _parts = cluster_name.split(" | ", 1)
        _plain_role = _parts[1] if len(_parts) == 2 else cluster_name
        country_part = _parts[0] if len(_parts) == 2 else "US"

        hiring_history = []
        cur.execute("""
            SELECT quarter, SUM(hire_count) as hires
            FROM cluster_hiring_history
            WHERE cluster_name = %s AND company_name = '_ALL_'
            GROUP BY quarter
            ORDER BY quarter DESC
            LIMIT 12
        """, (_plain_role,))
        hist_rows = cur.fetchall()
        if not hist_rows:
            # Try with the full cluster_name (may have country prefix)
            cur.execute("""
                SELECT quarter, SUM(hire_count) as hires
                FROM cluster_hiring_history
                WHERE cluster_name = %s AND company_name = '_ALL_'
                GROUP BY quarter
                ORDER BY quarter DESC
                LIMIT 12
            """, (cluster_name,))
            hist_rows = cur.fetchall()
        hiring_history = [{"quarter": r["quarter"], "hires": int(r["hires"])} for r in reversed(hist_rows)]

        # ─── Job Decomposition (top job titles) ───
        # For actual quarters, pull live data from updated_job_records
        job_decomp = []
        _quarter_decomp_used = False
        if quarter:
            import re as _re_q
            m_q = _re_q.match(r'Q(\d)\s*(\d{4})', quarter)
            if m_q:
                q_num, q_year = int(m_q.group(1)), int(m_q.group(2))
                q_start = f"{q_year}-{(q_num-1)*3+1:02d}-01"
                q_end = f"{q_year}-{q_num*3+1:02d}-01" if q_num < 4 else f"{q_year+1}-01-01"
                try:
                    cur.execute("""
                        SELECT title, COUNT(*) as cnt, SUM(openings) as opens
                        FROM updated_job_records
                        WHERE role_cluster = %s AND country = %s
                          AND issue_date >= %s AND issue_date < %s
                        GROUP BY title
                        ORDER BY opens DESC
                        LIMIT 12
                    """, (_plain_role, country_part if country_part else 'US', q_start, q_end))
                    qd_rows = cur.fetchall()
                    if qd_rows:
                        _quarter_decomp_used = True
                        # Get per-title skills from jd_extracted_skills
                        _db_quarter = f"{q_year}Q{q_num}"
                        _title_skills = {}
                        cur.execute("""
                            SELECT title, extracted_skills
                            FROM jd_extracted_skills
                            WHERE cluster_name = %s AND quarter = %s
                        """, (cluster_name, _db_quarter))
                        for es_row in cur.fetchall():
                            t = es_row["title"]
                            if t not in _title_skills:
                                _title_skills[t] = {}
                            for sk in json.loads(es_row["extracted_skills"]) if es_row["extracted_skills"] else []:
                                _title_skills[t][sk] = _title_skills[t].get(sk, 0) + 1

                        import re as _re_fz

                        # First pass: collect raw decomposition
                        _raw_decomp = []
                        for dr in qd_rows:
                            title = dr["title"]
                            skills_for_title = []
                            if title in _title_skills:
                                skills_for_title = [s for s, _ in sorted(_title_skills[title].items(), key=lambda x: -x[1])[:5]]
                            else:
                                # Fuzzy match: strip trailing variant letters/numbers
                                base = _re_fz.sub(r'\s*[-]\s*[A-Z0-9]$', '', title).strip()
                                for t_key, t_skills in _title_skills.items():
                                    t_base = _re_fz.sub(r'\s*[-]\s*[A-Z0-9]$', '', t_key).strip()
                                    if t_base == base:
                                        skills_for_title = [s for s, _ in sorted(t_skills.items(), key=lambda x: -x[1])[:5]]
                                        break
                            _raw_decomp.append({
                                "title": title,
                                "count": int(dr["opens"]),
                                "skills": skills_for_title,
                            })

                        # Second pass: group variants with identical skills
                        _grouped = {}
                        for item in _raw_decomp:
                            base = _re_fz.sub(r'\s*[-]\s*[A-Z0-9]$', '', item["title"]).strip()
                            sk_key = tuple(item["skills"])
                            group_key = (base, sk_key)
                            if group_key in _grouped and sk_key:
                                _grouped[group_key]["count"] += item["count"]
                            else:
                                _grouped[group_key] = {
                                    "title": base if sk_key and base != item["title"] else item["title"],
                                    "count": item["count"],
                                    "skills": item["skills"],
                                }
                                # If first entry had variant suffix, use base name
                                if sk_key and base != item["title"]:
                                    _grouped[group_key]["title"] = base

                        job_decomp = sorted(_grouped.values(), key=lambda x: -x["count"])
                except Exception:
                    pass

        # Fallback to static cluster_job_decomposition
        if not _quarter_decomp_used:
            # For predicted quarters, use most recent actual quarter's decomposition as projected mix
            _is_predicted_quarter = False
            if quarter and has_predictions:
                import re as _re_pq
                m_pq = _re_pq.match(r'Q(\d)\s*(\d{4})', quarter)
                if m_pq:
                    _pq_num, _pq_year = int(m_pq.group(1)), int(m_pq.group(2))
                    _pq_start = f"{_pq_year}-{(_pq_num-1)*3+1:02d}-01"
                    _pq_end = f"{_pq_year}-{_pq_num*3+1:02d}-01" if _pq_num < 4 else f"{_pq_year+1}-01-01"
                    # Check if there are actual records in this quarter
                    cur.execute("""
                        SELECT COUNT(*) as cnt FROM updated_job_records
                        WHERE role_cluster = %s AND country = %s
                          AND issue_date >= %s AND issue_date < %s
                    """, (_plain_role, country_part if country_part else 'US', _pq_start, _pq_end))
                    if cur.fetchone()['cnt'] == 0:
                        _is_predicted_quarter = True

            if _is_predicted_quarter:
                # Fetch predicted skills from the skills forecasting model
                cur.execute("""
                    SELECT predicted_skills, skill_categories, confidence
                    FROM predicted_cluster_skills
                    WHERE cluster_name = %s AND quarter = %s
                    ORDER BY created_at DESC LIMIT 1
                """, (cluster_name, f"{_pq_year}Q{_pq_num}"))
                _pred_row = cur.fetchone()
                if not _pred_row:
                    # Try with plain role
                    _pred_quarter_key = f"{_pq_year}Q{_pq_num}"
                    cur.execute("""
                        SELECT predicted_skills, skill_categories, confidence
                        FROM predicted_cluster_skills
                        WHERE cluster_name = %s AND quarter = %s
                        ORDER BY created_at DESC LIMIT 1
                    """, (cluster_name, _pred_quarter_key))
                    _pred_row = cur.fetchone()
                if _pred_row:
                    _forecasted_skills = _parse_json_field(_pred_row['predicted_skills']) or []
                    skills = _forecasted_skills  # Override topSkills with forecasted
                    skills_intelligence = _normalize_skills_intelligence(
                        _parse_json_field(_pred_row.get('skill_categories'))
                    )
                    _nlp_skills_note = f"Skills forecast by SkillForecast model (confidence: {_pred_row['confidence']})"
                else:
                    _nlp_skills_note = "Predicted quarter — no skill forecast available"

                # ─── Projected role mix for predicted quarters ───
                # Frontend already expects a `projected: true` flag on job_decomp
                # entries (see prediction-role-drilldown.component.ts) but the
                # backend never populated it. Fix: take the most recent actual
                # quarter's real role/skill breakdown for this cluster and scale
                # each role's count proportionally to this quarter's forecasted
                # total, so recruiters see a believable role-level breakdown
                # instead of only flat cluster-wide skills.
                try:
                    cur.execute("""
                        SELECT issue_date FROM updated_job_records
                        WHERE role_cluster = %s AND country = %s
                        ORDER BY issue_date DESC LIMIT 1
                    """, (_plain_role, country_part if country_part else 'US'))
                    _last_row = cur.fetchone()
                    if _last_row and _last_row.get('issue_date'):
                        _last_date = _last_row['issue_date']
                        _last_q_num = (_last_date.month - 1) // 3 + 1
                        _last_q_year = _last_date.year
                        _last_q_start = f"{_last_q_year}-{(_last_q_num-1)*3+1:02d}-01"
                        _last_q_end = f"{_last_q_year}-{_last_q_num*3+1:02d}-01" if _last_q_num < 4 else f"{_last_q_year+1}-01-01"

                        # ─── Company-scoped role mix: try company-specific first, fall back to industry ───
                        _mix_rows = []
                        _mix_is_company_specific = False
                        if company:
                            cur.execute("""
                                SELECT title, COUNT(*) as cnt, SUM(openings) as opens
                                FROM updated_job_records
                                WHERE role_cluster = %s AND country = %s
                                  AND issue_date >= %s AND issue_date < %s
                                  AND company_name LIKE %s
                                GROUP BY title
                                ORDER BY opens DESC
                                LIMIT 8
                            """, (_plain_role, country_part if country_part else 'US', _last_q_start, _last_q_end, f"%{company}%"))
                            _company_mix = cur.fetchall()
                            _company_mix_opens = sum(int(r['opens']) for r in _company_mix) if _company_mix else 0
                            if len(_company_mix) >= 3 and _company_mix_opens >= 5:
                                _mix_rows = _company_mix
                                _mix_is_company_specific = True
                            # else: too little data — fall through to full-cluster query

                        if not _mix_rows:
                            cur.execute("""
                                SELECT title, COUNT(*) as cnt, SUM(openings) as opens
                                FROM updated_job_records
                                WHERE role_cluster = %s AND country = %s
                                  AND issue_date >= %s AND issue_date < %s
                                GROUP BY title
                                ORDER BY opens DESC
                                LIMIT 8
                            """, (_plain_role, country_part if country_part else 'US', _last_q_start, _last_q_end))
                            _mix_rows = cur.fetchall()
                            _mix_is_company_specific = False

                        if _mix_rows:
                            _mix_total = sum(int(r['opens']) for r in _mix_rows) or 1
                            # predicted_demand is this endpoint's sum of demand_predicted
                            # for the requested quarter — exactly the "Q3 2026 DEMAND" KPI
                            # shown in the drilldown header, so scale the role mix to that.
                            _target_total = int(predicted_demand) if predicted_demand else _mix_total

                            # ─── Build per-role skill blends ───
                            # Pre-load ALL market signals once (not per-role) then match by relevant_roles
                            cur.execute("""
                                SELECT skill_name, market_trend, relevant_roles
                                FROM market_skill_signals
                                ORDER BY market_demand_score DESC
                            """)
                            _all_market_signals = cur.fetchall()

                            def _blend_role_skills(title):
                                """Build a weighted, role-specific predicted skill list:
                                - Historical: jd_extracted_skills matched by exact title, 3 quarters weighted 3/2/1
                                - Market boost: market_skill_signals where relevant_roles contains this title
                                Returns list of skill strings, most relevant first, or None.
                                """
                                cur2 = conn.cursor(dictionary=True)
                                cur2.execute("""
                                    SELECT quarter, extracted_skills FROM jd_extracted_skills
                                    WHERE title = %s
                                    ORDER BY quarter DESC LIMIT 3
                                """, (title,))
                                _hist_rows = cur2.fetchall()
                                cur2.close()

                                weights = [3, 2, 1]
                                skill_scores = {}
                                for w, hr in zip(weights, _hist_rows):
                                    try:
                                        sk = json.loads(hr["extracted_skills"]) if isinstance(hr["extracted_skills"], str) else (hr["extracted_skills"] or [])
                                    except Exception:
                                        sk = []
                                    for s in sk:
                                        if s and len(s) > 2:
                                            skill_scores[s] = skill_scores.get(s, 0) + w

                                # Market signal boost — match by relevant_roles containing this title
                                title_lower = title.lower()
                                for sig in _all_market_signals:
                                    try:
                                        roles_list = json.loads(sig["relevant_roles"]) if isinstance(sig["relevant_roles"], str) else (sig["relevant_roles"] or [])
                                    except Exception:
                                        roles_list = []
                                    role_match = any(
                                        title_lower in rl.lower() or rl.lower() in title_lower
                                        or any(word in title_lower for word in rl.lower().split() if len(word) > 4)
                                        for rl in roles_list
                                    )
                                    if role_match:
                                        sname = sig["skill_name"]
                                        boost = 2.0 if (sig.get("market_trend") or "").lower() == "hot" else 1.5
                                        skill_scores[sname] = skill_scores.get(sname, 0) + boost

                                if not skill_scores:
                                    return None  # caller falls back to cluster-level
                                ranked = sorted(skill_scores.items(), key=lambda x: -x[1])
                                return [s for s, _ in ranked[:6]]

                            job_decomp = []
                            _allocated = 0
                            for i, r in enumerate(_mix_rows):
                                share = int(r['opens']) / _mix_total
                                projected_count = max(1, round(_target_total * share)) if i < len(_mix_rows) - 1 else max(1, _target_total - _allocated)
                                _allocated += projected_count
                                role_skills = _blend_role_skills(r['title'])
                                if not role_skills:
                                    role_skills = (skills or [])[:5]  # true last-resort
                                _, role_skill_cats = _skills_for_projected_role(
                                    role_skills, skills_intelligence
                                )
                                job_decomp.append({
                                    "title": r["title"],
                                    "count": projected_count,
                                    "skills": role_skills,
                                    "skillsByCategory": role_skill_cats,
                                    "projected": True,
                                    "roleMixSource": "company" if _mix_is_company_specific else "industry-wide (insufficient company-specific data)",
                                })
                except Exception:
                    pass
            else:
                # Actual quarter but no records matched (edge case) — use static fallback
                cur.execute("""
                    SELECT job_title, SUM(jd_count) as count, top_skills
                    FROM cluster_job_decomposition
                    WHERE cluster_name = %s
                    GROUP BY job_title, top_skills
                    ORDER BY count DESC
                    LIMIT 8
                """, (_plain_role,))
                decomp_rows = cur.fetchall()
                if not decomp_rows:
                    cur.execute("""
                        SELECT job_title, SUM(jd_count) as count, top_skills
                        FROM cluster_job_decomposition
                        WHERE cluster_name = %s
                        GROUP BY job_title, top_skills
                        ORDER BY count DESC
                        LIMIT 8
                    """, (cluster_name,))
                    decomp_rows = cur.fetchall()
                for dr in decomp_rows:
                    skills_list = json.loads(dr["top_skills"]) if dr.get("top_skills") else []
                    job_decomp.append({
                        "title": dr["job_title"],
                        "count": int(dr["count"]),
                        "skills": skills_list[:5],
                    })

        # ─── Skill Evolution (from skill_trends table) ───
        skill_evolution = []
        cur.execute("""
            SELECT skill, trend_category, trending_score, current_frequency, previous_frequency, first_seen
            FROM skill_trends
            WHERE cluster_name = %s AND company_name = '_ALL_'
            ORDER BY trending_score DESC
            LIMIT 12
        """, (_plain_role,))
        evo_rows = cur.fetchall()
        if not evo_rows:
            cur.execute("""
                SELECT skill, trend_category, trending_score, current_frequency, previous_frequency, first_seen
                FROM skill_trends
                WHERE cluster_name = %s AND company_name = '_ALL_'
                ORDER BY trending_score DESC
                LIMIT 12
            """, (cluster_name,))
            evo_rows = cur.fetchall()
        for er in evo_rows:
            if not _is_clean_skill(er["skill"]):
                continue
            skill_evolution.append({
                "skill": er["skill"],
                "category": er["trend_category"],
                "score": float(er["trending_score"]) if er["trending_score"] else 0,
                "recentFreq": int(er["current_frequency"]) if er["current_frequency"] else 0,
                "previousFreq": int(er["previous_frequency"]) if er["previous_frequency"] else 0,
                "firstSeen": er.get("first_seen"),
            })

        # ─── Market Context (industry hiring trends) ───
        # Source 1: skill_trends from top clients (not _ALL_) — what companies are asking for
        market_context = []
        cur.execute("""
            SELECT skill, trend_category, trending_score,
                   current_frequency, previous_frequency
            FROM skill_trends
            WHERE cluster_name = %s AND company_name != '_ALL_'
            ORDER BY trending_score DESC
            LIMIT 50
        """, (_plain_role,))
        client_trend_rows = cur.fetchall()
        if not client_trend_rows:
            cur.execute("""
                SELECT skill, trend_category, trending_score,
                       current_frequency, previous_frequency
                FROM skill_trends
                WHERE cluster_name = %s AND company_name != '_ALL_'
                ORDER BY trending_score DESC
                LIMIT 50
            """, (cluster_name,))
            client_trend_rows = cur.fetchall()

        # Aggregate client skill trends
        client_skill_scores = {}
        for cr in client_trend_rows:
            sk = cr["skill"]
            if not _is_clean_skill(sk):
                continue
            score = float(cr["trending_score"]) if cr["trending_score"] else 0
            freq = int(cr["current_frequency"]) if cr["current_frequency"] else 0
            prev = int(cr["previous_frequency"]) if cr["previous_frequency"] else 0
            if sk not in client_skill_scores:
                client_skill_scores[sk] = {"score": 0, "freq": 0, "prev": 0, "cat": cr["trend_category"]}
            client_skill_scores[sk]["score"] += score
            client_skill_scores[sk]["freq"] += freq
            client_skill_scores[sk]["prev"] += prev

        for sk, data in sorted(client_skill_scores.items(), key=lambda x: -x[1]["score"])[:6]:
            yoy = None
            if data["prev"] > 0:
                yoy = round(((data["freq"] - data["prev"]) / data["prev"]) * 100)
            trend_label = "hot" if data["cat"] == "rising" and data["score"] > 5 else \
                          "growing" if data["cat"] in ("rising", "emerging") else "stable"
            market_context.append({
                "skill": sk,
                "trend": trend_label,
                "score": round(data["score"], 1),
                "yoyGrowth": yoy,
                "category": data["cat"],
            })

        # Source 2: market_skill_signals for matching roles
        if not market_context:
            cur.execute("""
                SELECT skill_name, market_trend, market_demand_score, yoy_growth_pct, category
                FROM market_skill_signals
                WHERE JSON_CONTAINS(relevant_roles, %s)
                ORDER BY market_demand_score DESC
                LIMIT 6
            """, (json.dumps(_plain_role),))
            for mr in cur.fetchall():
                market_context.append({
                    "skill": mr["skill_name"],
                    "trend": mr["market_trend"],
                    "score": int(mr["market_demand_score"]),
                    "yoyGrowth": round(mr["yoy_growth_pct"]) if mr["yoy_growth_pct"] else None,
                    "category": mr["category"],
                })

        if skills_intelligence:
            market_block = skills_intelligence.get("market") or {}
            has_market = any(market_block.get(k) for k in ("hot", "growing", "stable"))
            if not has_market and market_context:
                # Backfill market from live skill_trends when stored data is sparse
                skills_intelligence["market"] = {
                    "hot": [{"skill": m["skill"], "score": m.get("score", 0), "category": m.get("category", "general")}
                            for m in market_context if m.get("trend") == "hot"][:6],
                    "growing": [{"skill": m["skill"], "score": m.get("score", 0), "category": m.get("category", "general")}
                                for m in market_context if m.get("trend") in ("growing", "hot")][:6],
                    "stable": [{"skill": m["skill"], "score": m.get("score", 0), "category": m.get("category", "general")}
                               for m in market_context if m.get("trend") == "stable"][:6],
                }
            elif has_market and market_context:
                # Supplement: add live trending skills not already in stored market data
                existing_market_skills = set()
                for k in ("hot", "growing", "stable"):
                    for item in (market_block.get(k) or []):
                        skill_name = item.get("skill", "") if isinstance(item, dict) else item
                        existing_market_skills.add(skill_name.lower())
                for mc in market_context:
                    if mc["skill"].lower() not in existing_market_skills:
                        trend = mc.get("trend", "stable")
                        entry = {"skill": mc["skill"], "score": mc.get("score", 0), "category": mc.get("category", "general")}
                        if trend == "hot" and len(market_block.get("hot", [])) < 8:
                            market_block.setdefault("hot", []).append(entry)
                        elif trend in ("growing", "hot") and len(market_block.get("growing", [])) < 8:
                            market_block.setdefault("growing", []).append(entry)
                        elif len(market_block.get("stable", [])) < 8:
                            market_block.setdefault("stable", []).append(entry)
                        existing_market_skills.add(mc["skill"].lower())

        # Build skillsIntelligence for actual quarters from available data
        if not skills_intelligence and (skill_evolution or market_context):
            actual_core = [s for s in (skills or [])[:8] if s]
            actual_trending = [e["skill"] for e in skill_evolution if e.get("category") in ("rising", "emerging") and _is_clean_skill(e["skill"])][:6]
            actual_stable = [e["skill"] for e in skill_evolution if e.get("category") == "stable" and _is_clean_skill(e["skill"])][:6]
            skills_intelligence = {
                "predicted": {
                    "foundation": actual_core,
                    "trending_up": actual_trending,
                    "emerging": [],
                    "stable": actual_stable,
                },
                "historical": {
                    "core": actual_core,
                    "recent": [],
                    "momentum": actual_trending,
                    "stable": actual_stable,
                },
                "market": {
                    "hot": [{"skill": m["skill"], "score": m.get("score", 0), "category": m.get("category", "general")}
                            for m in market_context if m.get("trend") == "hot"][:6],
                    "growing": [{"skill": m["skill"], "score": m.get("score", 0), "category": m.get("category", "general")}
                                for m in market_context if m.get("trend") in ("growing", "hot")][:6],
                    "stable": [{"skill": m["skill"], "score": m.get("score", 0), "category": m.get("category", "general")}
                               for m in market_context if m.get("trend") == "stable"][:6],
                },
            }

        # Parse cluster into region/role
        parts = cluster_name.split(" | ", 1)
        country = parts[0] if len(parts) == 2 else ""
        role = parts[1] if len(parts) == 2 else cluster_name

        # Build summary text
        if predicted_demand > 0:
            summary_text = f"Plan for ~{predicted_demand} forecasted hires ({cluster_confidence} confidence)"
        elif estimated_demand > 0:
            summary_text = f"Trend estimate: ~{estimated_demand} hires (Jul-Dec), based on {actual_demand} recorded in Jan-May"
        else:
            summary_text = f"{actual_demand} hires recorded (actual data only)"

        # ─── Company filter: build an honest scope note ───
        # Actual scaling already happened once, up front, applied consistently
        # to both the per-month table and these aggregates — nothing left to
        # rescale here. This just labels what the person is looking at.
        scope_note = None
        if company:
            if _co_share_found:
                scope_note = f"Scoped to {company}"
            else:
                # We couldn't find company-specific data for this exact cluster —
                # say so plainly rather than silently showing full cluster
                # totals under a misleading "Scoped to X" label.
                scope_note = f"No {company}-specific data for this cluster — showing full cluster totals"

        cur.close()
    finally:
        conn.close()

    return {
        "cluster": cluster_name,
        "country": country,
        "region": get_cluster_region(cluster_name),
        "role": role,
        "mape": mape,
        "mae": float(meta_row["mae"]) if meta_row["mae"] else None,
        "mase": mase,
        "accuracy": round(100 - mape, 1) if mape else None,
        "grade": grade,
        "confidence": cluster_confidence,
        "reliable": True if not has_predictions else bool(meta_row["is_reliable"]),
        "hasForecasts": has_predictions,
        "scopeNote": scope_note,
        "topSkills": clean_skills(skills),
        "skillsSource": _nlp_skills_note,
        "skillsIntelligence": skills_intelligence,
        "topLocations": normalize_locations(locations),
        "topClients": raw_clients,
        "totalDemand": total_demand,
        "actualDemand": actual_demand,
        "predictedDemand": predicted_demand,
        "estimatedDemand": estimated_demand,
        "summary": summary_text,
        "predictionBasis": f"Model: {meta_row.get('model_used', 'HierBayes_v1_JOLTS')} | {len(actual_rows)} months actuals + {len(predicted_rows)} months forecast",
        "lastRefreshed": "Jul 2026",
        "predictions": predictions,
        "hiringHistory": hiring_history,
        "jobDecomposition": job_decomp,
        "skillEvolution": skill_evolution,
        "marketContext": market_context,
    }


@router.get("/skills")
def get_skill_demand(
    quarter: Optional[str] = Query(None, description="e.g. 2026-Q3"),
    region: Optional[str] = Query(None),
    company: Optional[str] = Query(None),
):
    """Skills ranked by weighted demand — tells recruiters what to prioritize."""
    conn = db_pool.get_connection()
    try:
        cur = conn.cursor(dictionary=True)

        # Build WHERE clause
        conditions = ["is_reliable = TRUE"]
        params = []

        if quarter:
            month_range = _quarter_to_month_range(quarter)
            if month_range:
                conditions.append("forecast_month BETWEEN %s AND %s")
                params.extend(month_range)

        if region:
            # Region is encoded in cluster_name prefix (US, IN, etc.). The frontend
            # now offers a single combined "LATAM & Others" option instead of
            # separate choices, so treat anything that isn't exactly "US" or
            # "India" as "not US and not India" rather than a hardcoded list.
            if region == "US":
                conditions.append("cluster_name LIKE %s")
                params.append("US |%")
            elif region == "India":
                conditions.append("cluster_name LIKE %s")
                params.append("IN |%")
            else:
                conditions.append("cluster_name NOT LIKE %s AND cluster_name NOT LIKE %s")
                params.extend(["US |%", "IN |%"])

        where_clause = " AND ".join(conditions)

        # If company filter, restrict to relevant clusters
        company_clusters = None
        if company:
            cur.execute("""
                SELECT DISTINCT cluster_name
                FROM company_cluster_profiles
                WHERE company_name LIKE %s
            """, (f"%{company}%",))
            company_clusters = set(r["cluster_name"] for r in cur.fetchall())

        cur.execute(f"""
            SELECT cluster_name, top_skills, SUM(demand_predicted) as total_demand
            FROM demand_forecasts
            WHERE {where_clause}
            GROUP BY cluster_name, top_skills
        """, params)
        rows = cur.fetchall()
        cur.close()
    finally:
        conn.close()

    skill_demand = {}
    for row in rows:
        # Filter by company clusters if applicable
        if company_clusters is not None and row["cluster_name"] not in company_clusters:
            continue

        skills = json.loads(row["top_skills"]) if row["top_skills"] else []
        skills = clean_skills(skills)
        demand = int(row["total_demand"])
        for skill in skills:
            if skill not in skill_demand:
                skill_demand[skill] = {"skill": skill, "weightedDemand": 0, "clusters": []}
            skill_demand[skill]["weightedDemand"] += demand
            skill_demand[skill]["clusters"].append(row["cluster_name"])

    ranked = sorted(skill_demand.values(), key=lambda x: x["weightedDemand"], reverse=True)
    return ranked[:30]


@router.get("/year-summary")
def get_year_summary():
    """Full year at-a-glance — constant, never changes with filters.
    
    Returns the executive summary: actuals (Jan-Jun) + predictions (Jul-Dec)
    with per-quarter breakdown. Used for the fixed KPI header.
    """
    conn = db_pool.get_connection()
    try:
        cur = conn.cursor(dictionary=True)

        # Per-quarter breakdown
        cur.execute("""
            SELECT 
                CASE 
                    WHEN forecast_month BETWEEN '2026-01' AND '2026-03' THEN '2026-Q1'
                    WHEN forecast_month BETWEEN '2026-04' AND '2026-06' THEN '2026-Q2'
                    WHEN forecast_month BETWEEN '2026-07' AND '2026-09' THEN '2026-Q3'
                    WHEN forecast_month BETWEEN '2026-10' AND '2026-12' THEN '2026-Q4'
                END as quarter,
                data_type,
                SUM(demand_predicted) as demand,
                COUNT(DISTINCT cluster_name) as clusters
            FROM demand_forecasts
            GROUP BY quarter, data_type
            ORDER BY quarter
        """)
        quarter_rows = cur.fetchall()

        # Reliable clusters count (mape <= 50, matching model's is_reliable flag)
        cur.execute("""
            SELECT COUNT(DISTINCT cluster_name) as cnt
            FROM demand_forecasts
            WHERE is_reliable = TRUE AND data_type = 'predicted'
        """)
        reliable_count = cur.fetchone()["cnt"]

        # WMAPE across all predicted clusters (weighted by demand)
        cur.execute("""
            SELECT SUM(demand_predicted * COALESCE(mape, 0)) / SUM(demand_predicted) as wmape
            FROM demand_forecasts
            WHERE data_type = 'predicted' AND mape IS NOT NULL
        """)
        wmape_row = cur.fetchone()
        avg_mape = float(wmape_row["wmape"]) if wmape_row and wmape_row["wmape"] else 38.0

        cur.close()
    finally:
        conn.close()

    # Build quarter summaries
    by_quarter = []
    actual_total = 0
    predicted_total = 0

    for row in quarter_rows:
        demand = int(row["demand"])
        q_entry = {
            "quarter": row["quarter"],
            "demand": demand,
            "clusters": row["clusters"],
            "dataType": row["data_type"],
        }
        by_quarter.append(q_entry)

        if row["data_type"] == "actual":
            actual_total += demand
        else:
            predicted_total += demand

    return {
        "year": 2026,
        "actualDemand": actual_total,
        "predictedDemand": predicted_total,
        "totalYear": actual_total + predicted_total,
        "byQuarter": by_quarter,
        "reliableClusters": reliable_count,
        "accuracy": round(100 - avg_mape, 1),
        "dataNote": "Jan-Jun = actual openings from JobDiva, Jul-Dec = ML forecast (Ensemble MaxAG)",
    }


@router.get("/summary")
def get_forecast_summary(
    quarter: Optional[str] = Query(None, description="e.g. 2026-Q3"),
    region: Optional[str] = Query(None),
    company: Optional[str] = Query(None),
):
    """KPI cards for dashboard header — respects region/company/quarter filters."""
    conn = db_pool.get_connection()
    try:
        cur = conn.cursor(dictionary=True)

        # Build WHERE conditions
        conditions = []
        params = []

        if quarter:
            month_range = _quarter_to_month_range(quarter)
            if month_range:
                conditions.append("forecast_month BETWEEN %s AND %s")
                params.extend(month_range)

        if region:
            # Same combined-region handling as the other endpoint above.
            if region == "US":
                conditions.append("cluster_name LIKE %s")
                params.append("US |%")
            elif region == "India":
                conditions.append("cluster_name LIKE %s")
                params.append("IN |%")
            else:
                conditions.append("cluster_name NOT LIKE %s AND cluster_name NOT LIKE %s")
                params.extend(["US |%", "IN |%"])

        # Company filter: restrict to clusters associated with this company
        company_clusters = None
        company_share_summary = {}  # cluster -> proportion for this company
        if company:
            cur.execute("""
                SELECT c.cluster_name, c.openings as company_openings,
                       t.total_openings, c.jd_count, t.total_jds
                FROM company_cluster_profiles c
                JOIN (
                    SELECT cluster_name, SUM(jd_count) as total_jds, SUM(openings) as total_openings
                    FROM company_cluster_profiles
                    GROUP BY cluster_name
                ) t ON c.cluster_name = t.cluster_name
                WHERE c.company_name LIKE %s
            """, (f"%{company}%",))
            company_rows = cur.fetchall()
            company_clusters = set()
            for cr in company_rows:
                cn = cr["cluster_name"]
                company_clusters.add(cn)
                if cr["total_openings"] and cr["total_openings"] > 0:
                    company_share_summary[cn] = cr["company_openings"] / cr["total_openings"]
                elif cr["total_jds"] and cr["total_jds"] > 0:
                    company_share_summary[cn] = cr["jd_count"] / cr["total_jds"]
                else:
                    company_share_summary[cn] = 1.0
            if company_clusters:
                placeholders = ",".join(["%s"] * len(company_clusters))
                conditions.append(f"cluster_name IN ({placeholders})")
                params.extend(list(company_clusters))

        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""

        # When company filter is active, compute proportional demand
        if company and company_share_summary:
            cur.execute(f"""
                SELECT cluster_name,
                       COUNT(DISTINCT forecast_month) as months,
                       SUM(demand_predicted) as cluster_demand,
                       MIN(forecast_month) as first_month,
                       MAX(forecast_month) as last_month
                FROM demand_forecasts
                {where_clause}
                GROUP BY cluster_name
            """, params)
            cluster_rows = cur.fetchall()
            total_clusters = len(cluster_rows)
            total_demand = 0
            first_month = None
            last_month = None
            for cr in cluster_rows:
                share = company_share_summary.get(cr["cluster_name"], 1.0)
                total_demand += max(1, round(cr["cluster_demand"] * share))
                if not first_month or cr["first_month"] < first_month:
                    first_month = cr["first_month"]
                if not last_month or cr["last_month"] > last_month:
                    last_month = cr["last_month"]
            summary = {
                "total_clusters": total_clusters,
                "total_demand": total_demand,
                "first_month": first_month,
                "last_month": last_month,
            }
        else:
            cur.execute(f"""
                SELECT
                    COUNT(DISTINCT cluster_name) as total_clusters,
                    SUM(demand_predicted) as total_demand,
                    MIN(forecast_month) as first_month,
                    MAX(forecast_month) as last_month
                FROM demand_forecasts
                {where_clause}
            """, params)
            summary = cur.fetchone()

        # Portfolio-level accuracy: validated on Q1 2026 holdout
        portfolio_mape = 7.6

        # Reliable clusters count (with same filters)
        reliable_conditions = conditions + ["is_reliable = TRUE", "mape <= 50"]
        reliable_where = "WHERE " + " AND ".join(reliable_conditions) if reliable_conditions else "WHERE is_reliable = TRUE AND mape <= 50"

        cur.execute(f"""
            SELECT COUNT(DISTINCT cluster_name) as reliable_clusters
            FROM demand_forecasts
            {reliable_where}
            GROUP BY cluster_name
            HAVING SUM(demand_predicted) >= 5
        """, params)
        reliable_rows = cur.fetchall()
        reliable_count = len(reliable_rows) if reliable_rows else 0

        # Top clusters in filtered set
        top_conditions = conditions + ["is_reliable = TRUE"]
        top_where = "WHERE " + " AND ".join(top_conditions) if top_conditions else "WHERE is_reliable = TRUE"

        cur.execute(f"""
            SELECT cluster_name, SUM(demand_predicted) as demand
            FROM demand_forecasts
            {top_where}
            GROUP BY cluster_name ORDER BY demand DESC LIMIT 5
        """, params)
        top_clusters = cur.fetchall()

        # Reliable demand
        cur.execute(f"""
            SELECT SUM(demand_predicted) as reliable_demand
            FROM demand_forecasts
            {"WHERE " + " AND ".join(conditions + ["is_reliable = TRUE"]) if conditions else "WHERE is_reliable = TRUE"}
        """, params)
        rel_row = cur.fetchone()
        reliable_demand = int(rel_row["reliable_demand"]) if rel_row and rel_row["reliable_demand"] else 0

        cur.close()
    finally:
        conn.close()

    total_demand_val = int(summary["total_demand"]) if summary and summary["total_demand"] else 1

    # Determine data type for this time window
    if quarter:
        month_range = _quarter_to_month_range(quarter)
        if month_range and month_range[1] <= "2026-06":
            summary_data_type = "actual"
        elif month_range and month_range[0] >= "2026-07":
            summary_data_type = "predicted"
        else:
            summary_data_type = "mixed"
    else:
        summary_data_type = "mixed"  # full year = both

    return {
        "forecastWindow": f"{summary['first_month']} to {summary['last_month']}" if summary and summary["first_month"] else "N/A",
        "totalClusters": summary["total_clusters"] if summary else 0,
        "reliableClusters": reliable_count,
        "totalDemand6Months": int(summary["total_demand"]) if summary and summary["total_demand"] else 0,
        "averageMape": round(portfolio_mape, 1),
        "overallAccuracy": round(100 - portfolio_mape, 1),
        "modelVersion": "HierBayes_v1_JOLTS",
        "dataType": summary_data_type,
        "demandCoverage": round(reliable_demand / total_demand_val * 100, 0) if total_demand_val > 0 else 0,
        "topClusters": [
            {"cluster": c["cluster_name"], "demand": int(c["demand"])}
            for c in top_clusters
        ],
    }