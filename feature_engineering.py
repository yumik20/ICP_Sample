"""
Step 3: Feature engineering on the consolidated (and optionally enriched) CSV.

Input:  lead_scoring_enriched.csv  (or lead_scoring_flat.csv if no enrichment)
Output: lead_scoring_features.csv  +  feature_summary.txt
"""

import pandas as pd
import re
from datetime import datetime

INPUT_FILE = "lead_scoring_enriched.csv"
FALLBACK_FILE = "lead_scoring_flat.csv"
OUTPUT_FILE = "lead_scoring_features.csv"
SUMMARY_FILE = "feature_summary.txt"

import os
input_path = INPUT_FILE if os.path.exists(INPUT_FILE) else FALLBACK_FILE
print(f"Using input: {input_path}")

df = pd.read_csv(input_path)
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

# ── 1. Seniority from Job Title ───────────────────────────────────────────────

SENIORITY_MAP = [
    # (regex pattern, level_name, numeric_score)
    (r"\b(ceo|cto|coo|cfo|cpo|cmo|chro|cso|president|founder|co-founder|owner|partner|managing director|executive director)\b",
     "c_suite", 5),
    (r"\b(svp|evp|vice president|vp)\b",
     "vp", 4),
    (r"\b(director|head of|principal|chief of staff)\b",
     "director", 3),
    (r"\b(senior manager|manager|lead|team lead|supervisor|superintendent)\b",
     "manager", 2),
    (r"\b(engineer|analyst|coordinator|associate|specialist|consultant|advisor|officer|generalist|recruiter|designer|developer|scientist|researcher|executive|assistant)\b",
     "ic", 1),
]

def parse_seniority(title):
    if not isinstance(title, str) or not title.strip():
        return "unknown", 0
    t = title.lower().strip()
    for pattern, level, score in SENIORITY_MAP:
        if re.search(pattern, t):
            return level, score
    return "unknown", 0

df[["seniority_label", "seniority_score"]] = df["job_title"].apply(
    lambda x: pd.Series(parse_seniority(x))
)

# ── 2. Company Size Normalization ─────────────────────────────────────────────

SIZE_RANGE_MAP = {
    "1-50": 25, "1 - 50": 25, "1-200": 100,
    "51-200": 125, "51 - 200": 125, "11-50": 30,
    "201-500": 350, "201 - 500": 350,
    "500-1000": 750, "501-1000": 750,
    "1000-5000": 2500, "1001-5000": 2500, "1k+": 2500,
    "5000-10000": 7500,
    "10000+": 15000, "10,000+": 15000,
}

def normalize_size(numeric_val, range_val):
    # Use numeric if available and sensible
    try:
        v = float(str(numeric_val).replace(",", ""))
        if 0 < v < 1_000_000:
            return v
    except (ValueError, TypeError):
        pass
    # Fall back to range midpoint
    if isinstance(range_val, str):
        key = range_val.strip().lower()
        for k, mid in SIZE_RANGE_MAP.items():
            if k.lower() == key:
                return float(mid)
        # Try to extract first number
        m = re.search(r"(\d[\d,]*)", range_val)
        if m:
            return float(m.group(1).replace(",", ""))
    return None

df["company_size"] = df.apply(
    lambda r: normalize_size(r["company_size_numeric"], r["company_size_range"]), axis=1
)

# Size bucket
def size_bucket(v):
    if pd.isna(v): return "unknown"
    if v <= 50: return "micro"
    if v <= 200: return "small"
    if v <= 500: return "mid"
    if v <= 1000: return "large"
    if v <= 5000: return "enterprise"
    return "mega"

df["size_bucket"] = df["company_size"].apply(size_bucket)

# ── 3. Industry Grouping ──────────────────────────────────────────────────────

INDUSTRY_MAP = [
    ("tech",        r"software|saas|tech|it service|information technology|cyber|ai |data|cloud|developer"),
    ("finance",     r"financ|bank|insurance|invest|capital|venture|asset management|account"),
    ("healthcare",  r"health|medic|pharma|biotech|hospital|clinical"),
    ("education",   r"educat|university|school|academ|e-learning|learning"),
    ("media",       r"media|marketing|advertis|pr |public relation|content|broadcast|audio|video|podcast"),
    ("manufacturing", r"manufactur|industrial|engineer|aerospace|automotive|hardware"),
    ("retail",      r"retail|ecommerce|e-commerce|consumer|fashion|food"),
    ("realestate",  r"real estate|property|construction|architecture"),
    ("government",  r"government|public sector|non.profit|npo|ngo|association"),
    ("hospitality", r"hotel|hospitality|travel|tourism|restaurant"),
    ("consulting",  r"consult|professional service|management"),
]

def map_industry(ind):
    if not isinstance(ind, str): return "other"
    t = ind.lower()
    for label, pattern in INDUSTRY_MAP:
        if re.search(pattern, t):
            return label
    return "other"

df["industry_group"] = df["industry"].apply(map_industry)

# ── 4. Platform Normalization ─────────────────────────────────────────────────

def norm_platform(p):
    if not isinstance(p, str): return "unknown"
    p = p.lower().strip()
    has_slack = "slack" in p
    has_teams = "teams" in p or "msteams" in p or "ms teams" in p
    if has_slack and has_teams: return "both"
    if has_slack: return "slack"
    if has_teams: return "teams"
    return "unknown"

df["platform_clean"] = df["platform"].apply(norm_platform)

# ── 5. Region Normalization ───────────────────────────────────────────────────

def norm_region(r):
    if not isinstance(r, str): return "unknown"
    r = r.lower().strip()
    if any(x in r for x in ("us", "usa", "united states", "canada", "north america")): return "north_america"
    if any(x in r for x in ("uk", "france", "germany", "spain", "italy", "netherlands",
                             "europe", "eu ", "eur", "sweden", "denmark", "norway",
                             "finland", "poland", "belgium", "austria", "switzerland")): return "europe"
    if any(x in r for x in ("australia", "new zealand", "apac", "asia", "japan",
                             "singapore", "india", "china", "korea")): return "apac"
    if any(x in r for x in ("brazil", "mexico", "latam", "latin america",
                             "colombia", "argentina", "chile")): return "latam"
    if any(x in r for x in ("africa", "nigeria", "kenya", "south africa", "middle east",
                             "uae", "saudi", "israel")): return "other"
    return "unknown"

df["region_clean"] = df["region"].apply(norm_region)

# ── 6. Engagement Duration ────────────────────────────────────────────────────

def parse_date(v):
    if not isinstance(v, str): return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(v[:10], fmt)
        except ValueError:
            continue
    return None

df["first_dt"] = df["first_contact_date"].apply(parse_date)
df["last_dt"] = df["last_contact_date"].apply(parse_date)
df["days_engaged"] = (df["last_dt"] - df["first_dt"]).dt.days
df["days_engaged"] = df["days_engaged"].apply(lambda x: max(0, x) if pd.notna(x) else None)

# Days since last contact (as of 2026-04-22)
REFERENCE_DATE = datetime(2026, 4, 22)
df["days_since_last_contact"] = df["last_dt"].apply(
    lambda d: (REFERENCE_DATE - d).days if d else None
)

# ── 7. Stage Score Features ───────────────────────────────────────────────────

def stage_bucket(score):
    if pd.isna(score): return "unknown"
    if score < 0: return "churned"
    if score == 0: return "dead"
    if score < 6: return "cold"
    if score < 8: return "warm"
    if score < 92: return "hot"
    return "converted"

df["stage_bucket"] = df["sale_stage_score"].apply(stage_bucket)

# ── 8. Derived Boolean Flags ──────────────────────────────────────────────────

df["is_enterprise"] = (
    (df["company_size"].fillna(0) >= 500) | (df["nda"].fillna(0) == 1)
).astype(int)

df["is_trial"] = df["trial"].fillna(0).astype(int)
df["has_nda"] = df["nda"].fillna(0).astype(int)

# ── 9. Payment Value Normalization ────────────────────────────────────────────

def to_float(v):
    try:
        return float(str(v).replace(",", "").replace("$", ""))
    except (ValueError, TypeError):
        return None

df["payment_2024_clean"] = df["payment_value_2024"].apply(to_float)
df["payment_2023_clean"] = df["payment_value_2023"].apply(to_float)

# ── 10. Save pre-encoded distributions for summary ───────────────────────────

pre_encode = {
    "seniority_label": df["seniority_label"].value_counts(),
    "industry_group":  df["industry_group"].value_counts(),
    "platform_clean":  df["platform_clean"].value_counts(),
    "region_clean":    df["region_clean"].value_counts(),
    "size_bucket":     df["size_bucket"].value_counts(),
    "stage_bucket":    df["stage_bucket"].value_counts(),
}

# ── 11. One-hot encode categoricals ──────────────────────────────────────────

df = pd.get_dummies(df, columns=["platform_clean", "industry_group", "region_clean",
                                  "seniority_label", "size_bucket", "stage_bucket"],
                    prefix=["plat", "ind", "reg", "sen", "sz", "stg"])

# ── Final feature set for modeling ───────────────────────────────────────────

FEATURE_COLS = [c for c in df.columns if c.startswith(
    ("plat_", "ind_", "reg_", "sen_", "sz_", "stg_")
)] + [
    "seniority_score",
    "company_size",
    "days_engaged",
    "days_since_last_contact",
    "is_enterprise",
    "is_trial",
    "has_nda",
    "sale_stage_score",
    "payment_2024_clean",
    "payment_2023_clean",
]

# Keep metadata columns too
META_COLS = ["sheet_origin", "job_title", "company", "industry", "platform",
             "region", "converted", "outcome_class", "sale_stage_raw"]

output_cols = META_COLS + [c for c in FEATURE_COLS if c in df.columns]
df_out = df[[c for c in output_cols if c in df.columns]]

df_out.to_csv(OUTPUT_FILE, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────

with open(SUMMARY_FILE, "w") as f:
    f.write("=== FEATURE ENGINEERING SUMMARY ===\n\n")
    f.write(f"Input rows: {len(df)}\n")
    f.write(f"Output columns: {len(df_out.columns)}\n\n")

    for label, dist in pre_encode.items():
        f.write(f"--- {label} distribution ---\n")
        for k, v in dist.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")

    f.write(f"\n--- Missing values (key features) ---\n")
    for col in ["company_size", "seniority_score", "days_engaged",
                "days_since_last_contact", "payment_2024_clean"]:
        if col in df.columns:
            missing = df[col].isna().sum()
            f.write(f"  {col}: {missing} missing ({100*missing/len(df):.1f}%)\n")

    f.write(f"\n--- Label distribution ---\n")
    f.write(str(df["outcome_class"].value_counts()) + "\n")
    f.write(f"\nConverted=1: {df['converted'].sum()} / {len(df)}\n")

print(f"Features written to: {OUTPUT_FILE}")
print(f"Summary written to:  {SUMMARY_FILE}")
print(f"\nFeature columns ({len(FEATURE_COLS)}):")
for c in FEATURE_COLS:
    if c in df.columns:
        print(f"  {c}")
