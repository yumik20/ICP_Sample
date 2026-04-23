"""
Step 2: Data enrichment — fill missing industry, company size, and region
using free/public lookup sources.

Strategy:
  1. Clearbit (free tier: 100/month) — best quality
  2. Apollo.io API (free tier)        — fallback
  3. Wikipedia / DuckDuckGo Instant   — last resort, no key needed

Set your API keys as environment variables before running:
  export CLEARBIT_API_KEY=your_key_here
  export APOLLO_API_KEY=your_key_here

If no keys are set, the script falls back to public lookup only.

Input:  lead_scoring_flat.csv
Output: lead_scoring_enriched.csv
        enrichment_log.csv
"""

import os
import csv
import json
import time
import re
import urllib.request
import urllib.parse
import urllib.error
from collections import defaultdict

INPUT_FILE  = "lead_scoring_flat.csv"
OUTPUT_FILE = "lead_scoring_enriched.csv"
LOG_FILE    = "enrichment_log.csv"
CACHE_FILE  = "enrichment_cache.json"

CLEARBIT_KEY = os.getenv("CLEARBIT_API_KEY", "")
APOLLO_KEY   = os.getenv("APOLLO_API_KEY", "")

# ── Load data ─────────────────────────────────────────────────────────────────

with open(INPUT_FILE, encoding="utf-8") as f:
    reader = csv.DictReader(f)
    fieldnames = reader.fieldnames[:]
    rows = list(reader)

print(f"Loaded {len(rows)} rows")

# ── Load cache ────────────────────────────────────────────────────────────────

cache = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE) as f:
        cache = json.load(f)
    print(f"Cache loaded: {len(cache)} companies")

def save_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)

# ── Normalize company name for cache key ──────────────────────────────────────

def normalize_company(name):
    if not name or not isinstance(name, str):
        return None
    name = name.strip().lower()
    name = re.sub(r"\b(inc|llc|ltd|corp|co|limited|group|gmbh|s\.a\.|plc|ag)\b\.?$", "", name)
    name = re.sub(r"[^a-z0-9 ]", "", name)
    return name.strip()

# ── Lookup sources ────────────────────────────────────────────────────────────

def fetch_clearbit(company_name):
    """Query Clearbit Company Name API."""
    if not CLEARBIT_KEY:
        return None
    url = f"https://company.clearbit.com/v2/companies/find?name={urllib.parse.quote(company_name)}"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {CLEARBIT_KEY}")
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
            return {
                "industry":  data.get("category", {}).get("industry"),
                "company_size_numeric": data.get("metrics", {}).get("employees"),
                "region":    data.get("geo", {}).get("country"),
                "source":    "clearbit",
            }
    except Exception as e:
        print(f"  Clearbit error for {company_name}: {e}")
        return None

def fetch_apollo(company_name):
    """Query Apollo.io organization search."""
    if not APOLLO_KEY:
        return None
    url = "https://api.apollo.io/v1/organizations/search"
    payload = json.dumps({"q_organization_name": company_name, "page": 1, "per_page": 1}).encode()
    req = urllib.request.Request(url, data=payload, method="POST")
    req.add_header("Content-Type", "application/json")
    req.add_header("Cache-Control", "no-cache")
    req.add_header("X-Api-Key", APOLLO_KEY)
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
            orgs = data.get("organizations", [])
            if not orgs:
                return None
            o = orgs[0]
            return {
                "industry":  o.get("industry"),
                "company_size_numeric": o.get("estimated_num_employees"),
                "region":    o.get("country"),
                "source":    "apollo",
            }
    except Exception as e:
        print(f"  Apollo error for {company_name}: {e}")
        return None

def fetch_duckduckgo(company_name):
    """DuckDuckGo Instant Answer — no API key, limited but free."""
    url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(company_name)}&format=json&no_html=1&skip_disambig=1"
    try:
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "LeadScoringEnricher/1.0")
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = json.loads(resp.read())
        # Extract any useful text from abstract
        abstract = data.get("AbstractText", "")
        infobox  = {item["label"]: item["value"]
                    for item in data.get("Infobox", {}).get("content", [])}
        employees = infobox.get("Number of employees") or infobox.get("Employees")
        hq = infobox.get("Headquarters") or infobox.get("Location")
        return {
            "industry": None,   # DDG doesn't return structured industry
            "company_size_numeric": employees,
            "region": hq,
            "source": "duckduckgo",
        } if (employees or hq) else None
    except Exception as e:
        print(f"  DDG error for {company_name}: {e}")
        return None

def enrich_company(company_name):
    key = normalize_company(company_name)
    if not key:
        return None
    if key in cache:
        return cache[key]

    result = (
        fetch_clearbit(company_name) or
        fetch_apollo(company_name) or
        fetch_duckduckgo(company_name)
    )
    cache[key] = result or {}
    return result

# ── Enrichment loop ───────────────────────────────────────────────────────────

log_rows = []
needs_enrichment_fields = ["industry", "company_size_numeric", "region"]

def needs_enrichment(row):
    return any(not row.get(f) for f in needs_enrichment_fields)

companies_seen = set()
enriched_count = 0
skipped_count  = 0

for i, row in enumerate(rows):
    company = row.get("company")
    key = normalize_company(company)

    if not company or not key:
        skipped_count += 1
        continue

    if not needs_enrichment(row) and key not in companies_seen:
        # Row already complete — but cache it for other rows with same company
        companies_seen.add(key)
        skipped_count += 1
        continue

    companies_seen.add(key)

    if i > 0 and i % 10 == 0:
        print(f"  Progress: {i}/{len(rows)} rows, {enriched_count} enriched...")
        save_cache()

    result = enrich_company(company)
    if not result:
        continue

    fields_filled = []

    if not row.get("industry") and result.get("industry"):
        row["industry"] = result["industry"]
        fields_filled.append("industry")

    if not row.get("company_size_numeric") and result.get("company_size_numeric"):
        row["company_size_numeric"] = result["company_size_numeric"]
        fields_filled.append("company_size_numeric")

    if not row.get("region") and result.get("region"):
        row["region"] = result["region"]
        fields_filled.append("region")

    if fields_filled:
        enriched_count += 1
        log_rows.append({
            "company": company,
            "source": result.get("source"),
            "fields_filled": ", ".join(fields_filled),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        })

    time.sleep(0.3)  # be polite to APIs

save_cache()

# ── Write output ──────────────────────────────────────────────────────────────

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

with open(LOG_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["company", "source", "fields_filled", "timestamp"])
    writer.writeheader()
    writer.writerows(log_rows)

print(f"\nDone.")
print(f"  Enriched: {enriched_count} rows")
print(f"  Skipped:  {skipped_count} rows (already complete or no company name)")
print(f"  Output:   {OUTPUT_FILE}")
print(f"  Log:      {LOG_FILE}")
print(f"  Cache:    {CACHE_FILE} ({len(cache)} entries)")

if not CLEARBIT_KEY and not APOLLO_KEY:
    print("\n*** No API keys set — only DuckDuckGo fallback was used.")
    print("    For better enrichment, set CLEARBIT_API_KEY or APOLLO_API_KEY.")
    print("    Clearbit free tier: https://clearbit.com/")
    print("    Apollo free tier:   https://apollo.io/")
