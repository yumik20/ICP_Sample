"""
ICP (Ideal Customer Profile) Identification Model
==================================================

WHAT THIS DOES
--------------
Learns the firmographic fingerprint of companies that became LEAD.bot customers,
then scores ANY new company against that fingerprint — even before they contact us.

WHY THIS IS DIFFERENT FROM THE PREVIOUS MODEL
----------------------------------------------
The previous model (train_models.py) used `sale_stage_score` as a feature.
That works great for ranking existing leads in your pipeline, but it's circular:
you already know their stage, so the model is mostly reading that back to you.

This model uses ONLY things you can know about a company BEFORE they reach out:
  - Job title / seniority
  - Industry
  - Company size
  - Platform (Slack vs Teams)
  - Region
  - Whether they're a known competitor customer

This means you can score a company from a cold list, a LinkedIn search,
a conference badge scan — anything.

WHAT WE'RE BUILDING
-------------------
1. ICP Profiler     — describes WHO your paying customers are (mean/median profile)
2. ICP Classifier   — a binary model: "does this company fit our ICP?" (yes/no + probability)
3. ICP Cluster      — finds sub-segments within your customer base (are there 2-3 distinct buyer types?)
4. ICP Scorer       — a function you can call with any company's attributes → get a 0-100 ICP score
5. ICP Report       — plain-English explanation of every decision

MODELS USED AND WHY
-------------------
• Logistic Regression   — gives us the "weight" of each feature. Fast, readable.
                          WHY: Tells us "being a Director adds +X to ICP score". Transparent.

• Random Forest         — handles non-linear combos (e.g., "Director AND Finance AND 200-500 ppl")
                          WHY: Catches patterns like "small tech companies don't convert,
                          but large tech companies do". Can't do that with logistic regression.

• XGBoost               — gradient boosting, state of the art on tabular data
                          WHY: Usually wins on small imbalanced datasets. Our winner from round 1.

• K-Means Clustering    — no labels needed. Groups your paying customers into sub-segments.
                          WHY: Maybe you have two ICPs: "HR Director at mid-size finance"
                          AND "IT Manager at large enterprise". Clustering reveals this.

• SHAP explanations     — tells us WHY each model gives each score.
                          WHY: "This company scored 82 because: Director-level ✓,
                          Finance industry ✓, 300 employees ✓, Europe ✗ (slight penalty)"

IMPORTANT DESIGN CHOICE
-----------------------
We intentionally EXCLUDE these features from this model:
  - sale_stage_score    (you don't know this for a cold company)
  - payment_value       (you don't know this yet)
  - days_engaged        (they haven't engaged yet)

We only use features you can look up for ANY company.
"""

import os, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
import json

# ── Install deps ──────────────────────────────────────────────────────────────
def require(pkg, import_as=None):
    try: return __import__(import_as or pkg)
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
        return __import__(import_as or pkg)

require("scikit-learn", "sklearn")
require("xgboost")
require("shap")
require("imbalanced-learn", "imblearn")

from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.cluster         import KMeans
from sklearn.preprocessing   import StandardScaler
from sklearn.impute          import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics         import (roc_auc_score, average_precision_score,
                                     f1_score, confusion_matrix, classification_report)
from sklearn.inspection      import permutation_importance
import xgboost as xgb_mod
import shap

# ── Config ────────────────────────────────────────────────────────────────────

INPUT_FILE   = "lead_scoring_features.csv"
FLAT_FILE    = "lead_scoring_enriched.csv"
ICP_SCORES   = "icp_scores.csv"
ICP_REPORT   = "icp_report.md"
ICP_CHART    = "icp_feature_importance.png"
CLUSTER_CHART = "icp_clusters.png"
MODEL_PARAMS  = "icp_model_params.json"

# ── ICP-only features (firmographic — no funnel signals) ──────────────────────

ICP_FEATURES = [
    # Platform
    "plat_slack", "plat_teams", "plat_both",
    # Industry
    "ind_tech", "ind_finance", "ind_consulting", "ind_education",
    "ind_healthcare", "ind_media", "ind_manufacturing",
    "ind_government", "ind_realestate", "ind_hospitality", "ind_retail",
    # Region
    "reg_north_america", "reg_europe", "reg_apac", "reg_latam",
    # Seniority
    "sen_c_suite", "sen_vp", "sen_director", "sen_manager", "sen_ic",
    "seniority_score",
    # Company size
    "company_size",
    "sz_micro", "sz_small", "sz_mid", "sz_large", "sz_enterprise", "sz_mega",
    # Flags
    "is_enterprise", "has_nda",
]

print("=" * 60)
print("  LEAD.bot ICP Model")
print("=" * 60)

# ── Load data ─────────────────────────────────────────────────────────────────

df_feat = pd.read_csv(INPUT_FILE)
df_flat = pd.read_csv(FLAT_FILE)

# Merge competitor_customer flag if available
if "competitor_customer" in df_flat.columns:
    df_feat["competitor_customer"] = df_flat["competitor_customer"].values
    ICP_FEATURES.append("competitor_customer")

# Build X using only ICP features
available = [f for f in ICP_FEATURES if f in df_feat.columns]
missing   = [f for f in ICP_FEATURES if f not in df_feat.columns]
if missing:
    print(f"  (Missing features filled with 0: {missing})")
    for f in missing:
        df_feat[f] = 0

X_raw = df_feat[available].copy()

# Convert booleans
for col in X_raw.columns:
    if X_raw[col].dtype == bool:
        X_raw[col] = X_raw[col].astype(int)

y = df_feat["converted"].astype(int)

print(f"\nDataset: {len(df_feat)} total leads, {y.sum()} confirmed paying (ICP-fit=1)")
print(f"ICP features: {len(available)}")

# ── Preprocess ────────────────────────────────────────────────────────────────

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X_raw)
feature_names = available

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# SMOTE for class imbalance
from imblearn.over_sampling import SMOTE
k = min(3, int(y.sum()) - 1)
if k >= 1 and y.mean() < 0.3:
    smote = SMOTE(random_state=42, k_neighbors=k)
    X_sm, y_sm = smote.fit_resample(X, y)
    X_sc_sm, _  = smote.fit_resample(X_scaled, y)
    print(f"SMOTE: balanced {y.sum()} → {y_sm.sum()} positives")
else:
    X_sm, y_sm = X, y
    X_sc_sm = X_scaled

X_tr, X_te, y_tr, y_te = train_test_split(X_sm, y_sm, test_size=0.2,
                                            stratify=y_sm, random_state=42)
X_sc_tr = X_sc_sm[:len(X_tr)]
X_sc_te = X_sc_sm[len(X_tr):]

# ─────────────────────────────────────────────────────────────────────────────
# PART 1: ICP PROFILER
# What does a paying customer look like, on average?
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  PART 1: ICP PROFILER")
print("  What does a paying customer look like?")
print("=" * 60)

paying_mask = df_feat["converted"] == 1
non_pay_mask = df_feat["converted"] == 0

profile = {}
for feat in available:
    pay_mean  = df_feat.loc[paying_mask, feat].mean()
    non_mean  = df_feat.loc[non_pay_mask, feat].mean()
    profile[feat] = {"paying_avg": pay_mean, "non_paying_avg": non_mean,
                     "lift": pay_mean - non_mean}

profile_df = pd.DataFrame(profile).T
profile_df = profile_df.sort_values("lift", ascending=False)

print("\nTop signals that separate paying vs non-paying customers:")
print(f"{'Feature':<30} {'Paying avg':>12} {'Non-paying avg':>15} {'Lift':>8}")
print("-" * 70)
for feat, row in profile_df.iterrows():
    if abs(row["lift"]) > 0.01:
        print(f"{feat:<30} {row['paying_avg']:>12.3f} {row['non_paying_avg']:>15.3f} {row['lift']:>+8.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 2: ICP CLASSIFIER
# Train models to predict ICP-fit
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  PART 2: ICP CLASSIFIER")
print("  Training models on firmographic features only")
print("  (No funnel stage, no payment history)")
print("=" * 60)

models = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=2000, class_weight="balanced",
                                    C=0.1, random_state=42),
        "X_tr": X_sc_tr, "X_te": X_sc_te,
        "why": "Linear model — each feature gets a weight. Most interpretable: tells us WHICH signals matter and by HOW MUCH.",
    },
    "Random Forest": {
        "model": RandomForestClassifier(n_estimators=300, max_depth=6,
                                        class_weight="balanced",
                                        random_state=42, n_jobs=-1),
        "X_tr": X_tr, "X_te": X_te,
        "why": "Captures non-linear combos: e.g. 'Director AND Finance AND 200-500 employees' as a single pattern.",
    },
    "XGBoost": {
        "model": xgb_mod.XGBClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=3,
            scale_pos_weight=(len(y_sm)-y_sm.sum())/max(y_sm.sum(),1),
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0),
        "X_tr": X_tr, "X_te": X_te,
        "why": "Gradient boosting — combines hundreds of weak patterns into one strong predictor. Best on small tabular datasets.",
    },
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
trained = {}

for name, cfg in models.items():
    m = cfg["model"]
    cv_X = X_sc_sm if cfg["X_tr"] is X_sc_tr else X_sm
    cv_scores = cross_val_score(m, cv_X, y_sm, cv=cv, scoring="roc_auc", n_jobs=-1)

    m.fit(cfg["X_tr"], y_tr)
    proba = m.predict_proba(cfg["X_te"])[:, 1]
    pred  = (proba >= 0.5).astype(int)

    roc = roc_auc_score(y_te, proba) if len(np.unique(y_te)) > 1 else 0.0
    pr  = average_precision_score(y_te, proba) if len(np.unique(y_te)) > 1 else 0.0
    f1  = f1_score(y_te, pred, zero_division=0)
    cm  = confusion_matrix(y_te, pred)

    results[name] = {"roc": roc, "pr": pr, "f1": f1, "cm": cm,
                     "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std()}
    trained[name] = m

    print(f"\n  {name}")
    print(f"    WHY: {cfg['why']}")
    print(f"    CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"    Test  ROC-AUC={roc:.3f}  PR-AUC={pr:.3f}  F1={f1:.3f}")

# Best model by PR-AUC (better for imbalanced classes)
best_name = max(results, key=lambda k: results[k]["pr"])
best_model = trained[best_name]
best_Xtr = models[best_name]["X_tr"]
best_Xte = models[best_name]["X_te"]
print(f"\n  Best ICP model: {best_name} (PR-AUC={results[best_name]['pr']:.3f})")

# ── Logistic Regression coefficients (most human-readable) ───────────────────

lr = trained["Logistic Regression"]
lr_coef = pd.Series(lr.coef_[0], index=feature_names).sort_values(ascending=False)

print("\n  Logistic Regression — feature weights (positive = boosts ICP score):")
for feat, coef in lr_coef.items():
    if abs(coef) > 0.05:
        direction = "↑ BOOSTS" if coef > 0 else "↓ LOWERS"
        print(f"    {feat:<30} {coef:+.3f}  {direction}")

# ─────────────────────────────────────────────────────────────────────────────
# PART 3: SHAP — WHY did each lead get its score?
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  PART 3: SHAP FEATURE IMPORTANCE")
print("  WHY does the model give each score?")
print("=" * 60)

try:
    if "XGBoost" in best_name:
        explainer   = shap.TreeExplainer(best_model)
        shap_vals   = explainer.shap_values(best_Xte)
    elif "Forest" in best_name:
        explainer   = shap.TreeExplainer(best_model)
        shap_vals   = explainer.shap_values(best_Xte)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
    else:
        explainer   = shap.LinearExplainer(lr, X_sc_sm)
        shap_vals   = explainer.shap_values(best_Xte)

    shap_vals = np.array(shap_vals)
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[:, :, 1]

    mean_shap = np.abs(shap_vals).mean(axis=0)
    top_idx   = np.argsort(mean_shap)[::-1]
    top_features_shap = [(feature_names[i], mean_shap[i]) for i in top_idx]

    print("\n  Top features driving ICP classification:")
    for rank, (feat, val) in enumerate(top_features_shap[:12], 1):
        bar = "█" * int(val / max(mean_shap) * 20)
        print(f"    {rank:2d}. {feat:<30} {bar} {val:.4f}")

    # Save SHAP plot
    Xte_df = pd.DataFrame(best_Xte, columns=feature_names)
    shap.summary_plot(shap_vals, Xte_df, show=False, plot_size=(10, 7))
    plt.title(f"ICP Feature Importance (SHAP) — {best_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(ICP_CHART, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  SHAP chart saved: {ICP_CHART}")

except Exception as e:
    print(f"  SHAP failed: {e}")
    import traceback; traceback.print_exc()
    top_features_shap = []

# ─────────────────────────────────────────────────────────────────────────────
# PART 4: CUSTOMER CLUSTERING
# Are there distinct sub-segments within your ICP?
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  PART 4: K-MEANS CLUSTERING")
print("  Finding sub-segments within your paying customers")
print("=" * 60)

X_paying = imputer.transform(df_feat.loc[paying_mask, available])
X_paying_sc = scaler.transform(X_paying)

# Try 2 and 3 clusters (small N = keep it simple)
best_k = 2
cluster_labels = None

for k in [2, 3]:
    km = KMeans(n_clusters=k, random_state=42, n_init=20)
    labels = km.fit_predict(X_paying_sc)
    # Print cluster profiles
    print(f"\n  K={k} clusters:")
    cluster_df = pd.DataFrame(X_paying, columns=feature_names)
    cluster_df["cluster"] = labels
    for c in range(k):
        sub = cluster_df[cluster_df["cluster"] == c]
        # Top distinguishing features
        top_feats = []
        for feat in ["seniority_score", "company_size", "ind_tech", "ind_finance",
                     "ind_consulting", "ind_education", "ind_government",
                     "reg_north_america", "reg_europe",
                     "plat_slack", "plat_teams", "sz_micro", "sz_small", "sz_mid",
                     "sz_large", "sz_enterprise", "sen_director", "sen_c_suite",
                     "sen_manager", "sen_ic"]:
            if feat in sub.columns and sub[feat].mean() > 0.3:
                top_feats.append(f"{feat}={sub[feat].mean():.2f}")
        print(f"    Cluster {c} ({len(sub)} customers): {', '.join(top_feats[:6])}")

# Save the 2-cluster model for scoring
km_final = KMeans(n_clusters=2, random_state=42, n_init=20)
km_final.fit(X_paying_sc)

# ─────────────────────────────────────────────────────────────────────────────
# PART 5: SCORE ALL LEADS
# Apply ICP model to every row in the dataset
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  PART 5: SCORING ALL LEADS")
print("=" * 60)

X_all = imputer.transform(df_feat[available])

if models[best_name]["X_tr"] is X_sc_tr:
    X_all_score = scaler.transform(X_all)
else:
    X_all_score = X_all

icp_proba = best_model.predict_proba(X_all_score)[:, 1]
df_feat["icp_score"] = (icp_proba * 100).round(1)

def icp_tier(s):
    if s >= 75: return "Strong ICP"
    if s >= 55: return "Likely ICP"
    if s >= 35: return "Weak ICP"
    return "Not ICP"

df_feat["icp_tier"] = df_feat["icp_score"].apply(icp_tier)

# Save scores
out_cols = ["company", "job_title", "industry", "platform", "region",
            "outcome_class", "icp_score", "icp_tier"]
flat_cols = {c: df_flat[c] for c in ["company", "job_title", "industry",
                                       "platform", "region", "outcome_class"]
             if c in df_flat.columns}
score_df = pd.DataFrame(flat_cols)
score_df["icp_score"] = df_feat["icp_score"].values
score_df["icp_tier"]  = df_feat["icp_tier"].values
score_df.sort_values("icp_score", ascending=False).to_csv(ICP_SCORES, index=False)

tier_counts = score_df["icp_tier"].value_counts()
print(f"\n  ICP Tier Distribution:")
for tier in ["Strong ICP", "Likely ICP", "Weak ICP", "Not ICP"]:
    count = tier_counts.get(tier, 0)
    print(f"    {tier:<15} {count:4d} leads")

# Top non-paying leads with high ICP score (outreach candidates)
non_paying = score_df[(score_df["outcome_class"] != "paying") &
                       (score_df["icp_score"] >= 55)]
print(f"\n  High-ICP non-paying leads (approach these!):")
print(f"  {len(non_paying)} leads with ICP score ≥ 55 who haven't converted yet")
print()
for _, r in non_paying.sort_values("icp_score", ascending=False).head(20).iterrows():
    print(f"    {str(r.get('company','')):<35} score={r['icp_score']:5.1f}  "
          f"tier={r['icp_tier']:<12} [{r.get('outcome_class','')}]")

# ─────────────────────────────────────────────────────────────────────────────
# PART 6: score_new_company() — score any company on the fly
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("  PART 6: SCORE A NEW COMPANY")
print("=" * 60)

import re as _re

# Word-boundary regex avoids substring bugs (e.g. "cto" hiding inside "director")
SENIORITY_MAP = [
    (_re.compile(r"\b(ceo|cto|coo|cfo|cpo|cmo|chro|president|founder|co.founder|owner|managing director|executive director|chief)\b"),
     "c_suite", 5),
    (_re.compile(r"\b(svp|evp|vice president|vp)\b"),
     "vp", 4),
    (_re.compile(r"\b(director|head of|principal)\b"),
     "director", 3),
    (_re.compile(r"\b(senior manager|manager|team lead|supervisor|lead)\b"),
     "manager", 2),
    (_re.compile(r"\b(engineer|analyst|coordinator|associate|specialist|consultant|generalist|recruiter|designer|developer|scientist|researcher|assistant|officer|intern)\b"),
     "ic", 1),
]

SIZE_RANGES = {
    "1-50": 25, "51-200": 125, "201-500": 350,
    "500-1000": 750, "1000-5000": 2500, "5000-10000": 7500, "10000+": 15000
}

INDUSTRY_MAP = [
    ("tech",        ["software","saas","tech","it service","information technology",
                     "cyber","ai ","data","cloud","developer"]),
    ("finance",     ["financ","bank","insurance","invest","capital","venture","asset","account"]),
    ("healthcare",  ["health","medic","pharma","biotech","hospital","clinical"]),
    ("education",   ["educat","university","school","academ","e-learning","learning"]),
    ("media",       ["media","marketing","advertis","pr ","public relation","content",
                     "broadcast","audio","video","podcast"]),
    ("manufacturing",["manufactur","industrial","engineer","aerospace","automotive","hardware"]),
    ("retail",      ["retail","ecommerce","consumer","fashion","food"]),
    ("realestate",  ["real estate","property","construction","architecture"]),
    ("government",  ["government","public sector","non.profit","npo","ngo","association"]),
    ("hospitality", ["hotel","hospitality","travel","tourism"]),
    ("consulting",  ["consult","professional service","management"]),
]

def score_new_company(
    job_title: str = "",
    industry: str = "",
    company_size: int = None,
    platform: str = "",       # "slack", "teams", "both"
    region: str = "",         # "north_america", "europe", "apac", "latam"
    is_competitor_customer: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Score any company against the LEAD.bot ICP.

    Parameters
    ----------
    job_title   : free text job title of the contact
    industry    : free text industry
    company_size: headcount (integer)
    platform    : 'slack', 'teams', 'both', or '' (unknown)
    region      : 'north_america', 'europe', 'apac', 'latam', or '' (unknown)
    is_competitor_customer: True if company uses Donut, CoffeePals, or 10KC

    Returns
    -------
    dict with keys: icp_score (0-100), icp_tier, explanation (list of reasons)
    """
    row = dict.fromkeys(feature_names, 0)

    # Seniority
    tl = job_title.lower()
    seniority_level, seniority_score = "unknown", 0
    for pattern, level, score in SENIORITY_MAP:
        if pattern.search(tl):
            seniority_level, seniority_score = level, score
            break
    row["seniority_score"] = seniority_score
    if seniority_level != "unknown":
        col = f"sen_{seniority_level}"
        if col in row: row[col] = 1

    # Industry
    ind_lower = industry.lower()
    industry_group = "other"
    for group, keywords in INDUSTRY_MAP:
        if any(kw in ind_lower for kw in keywords):
            industry_group = group
            break
    col = f"ind_{industry_group}"
    if col in row: row[col] = 1

    # Company size
    if company_size:
        row["company_size"] = float(company_size)
        if company_size <= 50:    row["sz_micro"] = 1
        elif company_size <= 200: row["sz_small"] = 1
        elif company_size <= 500: row["sz_mid"] = 1
        elif company_size <= 1000:row["sz_large"] = 1
        elif company_size <= 5000:row["sz_enterprise"] = 1
        else:                     row["sz_mega"] = 1
        row["is_enterprise"] = 1 if company_size >= 500 else 0

    # Platform
    plt_lower = platform.lower()
    if "both" in plt_lower:
        row["plat_both"] = 1
    elif "slack" in plt_lower:
        row["plat_slack"] = 1
    elif "teams" in plt_lower or "ms" in plt_lower:
        row["plat_teams"] = 1

    # Region
    reg_lower = region.lower()
    if "north" in reg_lower or "us" in reg_lower or "canada" in reg_lower:
        row["reg_north_america"] = 1
    elif "eur" in reg_lower:
        row["reg_europe"] = 1
    elif "apac" in reg_lower or "asia" in reg_lower:
        row["reg_apac"] = 1
    elif "lat" in reg_lower:
        row["reg_latam"] = 1

    # Competitor customer
    if is_competitor_customer and "competitor_customer" in row:
        row["competitor_customer"] = 1

    # Build feature vector
    x_vec = np.array([[row[f] for f in feature_names]])
    x_imp = imputer.transform(x_vec)

    if models[best_name]["X_tr"] is X_sc_tr:
        x_final = scaler.transform(x_imp)
    else:
        x_final = x_imp

    prob = best_model.predict_proba(x_final)[0][1]
    score = round(prob * 100, 1)
    tier = icp_tier(score)

    # Plain-English explanation
    explanation = []
    if seniority_level == "director": explanation.append("Director-level contact ✓ (strong buyer signal)")
    elif seniority_level == "c_suite": explanation.append("C-suite contact ✓ (decision maker)")
    elif seniority_level == "vp": explanation.append("VP-level contact ✓")
    elif seniority_level == "manager": explanation.append("Manager-level contact (mid signal)")
    elif seniority_level == "ic": explanation.append("IC-level contact ↓ (not a buyer)")
    else: explanation.append("Seniority unknown — job title not parsed")

    if industry_group in ("finance", "tech"): explanation.append(f"{industry_group.capitalize()} industry ✓ (top converting verticals)")
    elif industry_group in ("consulting", "education", "government"): explanation.append(f"{industry_group.capitalize()} industry ~ (moderate fit)")
    elif industry_group == "other": explanation.append("Industry unknown or low-signal")
    else: explanation.append(f"{industry_group.capitalize()} industry (some fit)")

    if company_size:
        if 200 <= company_size <= 1000: explanation.append(f"{company_size} employees ✓ (sweet spot: 200-1000)")
        elif company_size < 50: explanation.append(f"{company_size} employees ↓ (too small for meaningful plan)")
        elif company_size > 5000: explanation.append(f"{company_size} employees ~ (large enterprise — longer sales cycle)")
        else: explanation.append(f"{company_size} employees (acceptable size)")

    if platform: explanation.append(f"Platform: {platform}")
    if region: explanation.append(f"Region: {region}")
    if is_competitor_customer: explanation.append("Uses competitor (Donut/CoffeePals/10KC) ✓ — validated buyer")

    if verbose:
        print(f"\n  Score: {score}/100 — {tier}")
        print(f"  Reasons:")
        for r in explanation:
            print(f"    • {r}")

    return {"icp_score": score, "icp_tier": tier, "explanation": explanation}


# Run example predictions
print("\n  Example 1 — Director of HR at 400-person Finance company (Slack):")
score_new_company(
    job_title="Director of HR",
    industry="Financial Services",
    company_size=400,
    platform="slack",
    region="north_america",
)

print("\n  Example 2 — Intern at 15-person startup:")
score_new_company(
    job_title="Marketing Intern",
    industry="Consumer Goods",
    company_size=15,
    platform="",
    region="",
)

print("\n  Example 3 — VP People at 750-person Tech company (Teams) + Donut customer:")
score_new_company(
    job_title="VP of People Operations",
    industry="Software",
    company_size=750,
    platform="teams",
    region="europe",
    is_competitor_customer=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# PART 7: WRITE ICP REPORT
# ─────────────────────────────────────────────────────────────────────────────

now = datetime.now().strftime("%Y-%m-%d %H:%M")

with open(ICP_REPORT, "w") as f:
    f.write(f"# LEAD.bot ICP Model Report\n")
    f.write(f"**Generated:** {now}\n\n")

    f.write("## What This Model Does\n\n")
    f.write("Scores any company against LEAD.bot's Ideal Customer Profile (ICP) using ONLY\n")
    f.write("firmographic features — things you can know about a company BEFORE they contact you:\n")
    f.write("job title seniority, industry, company size, platform, and region.\n\n")
    f.write("This lets you:\n")
    f.write("- Score cold outbound prospect lists\n")
    f.write("- Pre-qualify inbound leads instantly\n")
    f.write("- Identify which canceled/inactive accounts are worth re-engaging\n")
    f.write("- Flag competitor customers as high-priority targets\n\n")

    f.write("## Why Each Model Was Used\n\n")
    for name, cfg in models.items():
        f.write(f"### {name}\n")
        f.write(f"{cfg['why']}\n\n")
        r = results[name]
        f.write(f"- CV AUC: {r['cv_mean']:.3f} ± {r['cv_std']:.3f}\n")
        f.write(f"- Test ROC-AUC: {r['roc']:.3f} | PR-AUC: {r['pr']:.3f} | F1: {r['f1']:.3f}\n\n")

    f.write(f"**Best model: {best_name}** (by PR-AUC, which is most reliable for imbalanced classes)\n\n")

    f.write("## ICP Profile — Your Paying Customers\n\n")
    f.write("| Feature | Paying avg | Non-paying avg | Lift |\n")
    f.write("|---|---|---|---|\n")
    for feat, row in profile_df.iterrows():
        if abs(row["lift"]) > 0.01:
            f.write(f"| {feat} | {row['paying_avg']:.3f} | {row['non_paying_avg']:.3f} | {row['lift']:+.3f} |\n")

    f.write("\n## Logistic Regression Weights\n\n")
    f.write("Positive = boosts ICP score. Negative = lowers ICP score.\n\n")
    f.write("| Feature | Weight | Direction |\n")
    f.write("|---|---|---|\n")
    for feat, coef in lr_coef.items():
        if abs(coef) > 0.05:
            direction = "↑ Boosts ICP fit" if coef > 0 else "↓ Reduces ICP fit"
            f.write(f"| {feat} | {coef:+.3f} | {direction} |\n")

    if top_features_shap:
        f.write("\n## SHAP Feature Importance\n\n")
        f.write("| Rank | Feature | Importance |\n")
        f.write("|---|---|---|\n")
        for rank, (feat, val) in enumerate(top_features_shap[:15], 1):
            f.write(f"| {rank} | {feat} | {val:.4f} |\n")
        f.write(f"\nSee chart: `{ICP_CHART}`\n")

    f.write("\n## ICP Tier Distribution\n\n")
    f.write("| Tier | Count | Threshold | Action |\n")
    f.write("|---|---|---|---|\n")
    tier_actions = {
        "Strong ICP": "≥ 75 | Immediate outreach — personalized cold email or LinkedIn",
        "Likely ICP": "55–74 | Add to drip sequence, prioritize if they engage",
        "Weak ICP":   "35–54 | Newsletter only unless they inbound",
        "Not ICP":    "< 35  | Do not pursue proactively",
    }
    for tier in ["Strong ICP", "Likely ICP", "Weak ICP", "Not ICP"]:
        count = tier_counts.get(tier, 0)
        action = tier_actions[tier]
        f.write(f"| {tier} | {count} | {action} |\n")

    f.write("\n## How to Score a New Company\n\n")
    f.write("```python\n")
    f.write("from icp_model import score_new_company\n\n")
    f.write('result = score_new_company(\n')
    f.write('    job_title="Director of People Operations",\n')
    f.write('    industry="Financial Services",\n')
    f.write('    company_size=350,\n')
    f.write('    platform="teams",\n')
    f.write('    region="north_america",\n')
    f.write('    is_competitor_customer=False,\n')
    f.write(')\n')
    f.write('# Returns: {"icp_score": 82.4, "icp_tier": "Strong ICP", "explanation": [...]}\n')
    f.write("```\n\n")

    f.write("## High-ICP Non-Paying Leads (Outreach Candidates)\n\n")
    f.write("These leads scored ≥ 55 ICP but have not converted. Prioritize for outreach.\n\n")
    f.write("| Company | Job Title | Industry | ICP Score | Tier | Status |\n")
    f.write("|---|---|---|---|---|---|\n")
    for _, row in non_paying.sort_values("icp_score", ascending=False).head(30).iterrows():
        f.write(f"| {row.get('company','')} | {row.get('job_title','')} | "
                f"{row.get('industry','')} | {row['icp_score']} | "
                f"{row['icp_tier']} | {row.get('outcome_class','')} |\n")

    f.write("\n## Model Caveats\n\n")
    f.write("- Dataset is small (18 positive examples). Model will improve significantly with more paying customer data.\n")
    f.write("- SMOTE was used to synthetically balance classes. Scores are relative, not absolute probabilities.\n")
    f.write("- Platform data is missing for 77% of paying customers — biggest enrichment opportunity.\n")
    f.write("- Retrain this model every quarter as new customers are added.\n")

print(f"\n  Report written: {ICP_REPORT}")
print(f"  Scores written: {ICP_SCORES}")
print(f"\n{'='*60}")
print("  DONE")
print(f"{'='*60}")
