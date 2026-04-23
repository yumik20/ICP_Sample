"""
ICP Model v2 — Based Purely on LEAD.bot's Own Data
====================================================

WHAT CHANGED FROM v1 AND WHY
------------------------------
v1 made assumptions influenced by competitor research (Teams=bad, Tech=bad).
Those signals were real in our data, but they reflect OUR current sales reach
and marketing message — not who actually has the problem we solve.

v2 is built around one question:
  "Who has the knowledge-silo problem that LEAD.bot solves?"

LEAD.bot's product solves: people at mid-size companies who need structured
knowledge sharing across departments or locations — onboarding, mentorship,
cross-functional pairing. This is different from Donut (social watercooler).

So our ICP isn't "social-first tech startups." It's:
  → Companies big enough to have silos
  → Where knowledge transfer is a structured business problem
  → With someone in charge of solving it (HR, People Ops, IT, Operations)
  → Who has budget authority (Director+)

NEW FEATURES WE ENGINEER
--------------------------
1. is_knowledge_industry  — industries where silos cause real business risk
                            (Finance, Gov, Consulting, Legal, Education)
2. is_hr_function         — job is explicitly about people programs
                            (HR, People, Culture, Talent, L&D)
3. is_ops_it_function     — job deploys the tool (IT, Ops, Systems)
4. is_silo_size           — 200-1500 employees (big enough for silos,
                            small enough to not have a full L&D team yet)
5. seniority_score        — Director+ has the problem + the budget

WHAT WE REMOVED
---------------
- competitor_customer: based on assumption, not our data
- platform features: reflect our sales reach, not the problem fit
- region features: reflect our marketing, not the problem fit

We keep region and platform as informational context, not model inputs.
"""

import os, warnings, re
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

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
from sklearn.preprocessing   import StandardScaler
from sklearn.impute          import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics         import (roc_auc_score, average_precision_score,
                                     f1_score, confusion_matrix)
from imblearn.over_sampling  import SMOTE
import xgboost as xgb_mod
import shap

FLAT_FILE  = "lead_scoring_flat.csv"
OUT_SCORES = "icp_v2_scores.csv"
OUT_REPORT = "icp_v2_report.md"

# ── Load raw flat data (before feature engineering) ───────────────────────────

df = pd.read_csv(FLAT_FILE)
print(f"Loaded {len(df)} rows")

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING — Problem-Driven
# Each feature is tied to the question: "Does this company have the silo problem?"
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Seniority — parsed with word-boundary regex ───────────────────────────
# WHY: Director+ recognizes the problem AND has budget to solve it.
#      ICs feel the pain but can't buy. C-suite delegates down.
#      Director is the sweet spot.

SENIORITY_RULES = [
    (re.compile(r"\b(ceo|cto|coo|cfo|cpo|cmo|chro|president|founder|co.founder|owner|managing director|executive director|chief)\b"), "c_suite", 5),
    (re.compile(r"\b(svp|evp|vice president|vp)\b"),                    "vp",        4),
    (re.compile(r"\b(director|head of|principal)\b"),                   "director",  3),
    (re.compile(r"\b(senior manager|manager|team lead|supervisor|lead)\b"), "manager", 2),
    (re.compile(r"\b(engineer|analyst|coordinator|associate|specialist|consultant|generalist|recruiter|designer|developer|scientist|researcher|assistant|officer|intern|partner)\b"), "ic", 1),
]

def parse_seniority(title):
    if not isinstance(title, str) or not title.strip():
        return "unknown", 0
    t = title.lower().strip()
    for pattern, level, score in SENIORITY_RULES:
        if pattern.search(t):
            return level, score
    return "unknown", 0

df[["seniority_label", "seniority_score"]] = df["job_title"].apply(
    lambda x: pd.Series(parse_seniority(x))
)

# ── 2. Job Function — HR/People vs IT/Ops vs Other ───────────────────────────
# WHY: HR and People Ops own the knowledge-sharing problem.
#      IT/Ops deploys and manages the tool.
#      These are the two buyer personas for LEAD.bot.

HR_PATTERN = re.compile(
    r"\b(hr|human resource|people|culture|talent|learning|l&d|"
    r"employee experience|engagement|org dev|organizational|"
    r"workforce|hrbp|people partner|staff development|diversity|dei|"
    r"recruitment|onboard)\b"
)
OPS_IT_PATTERN = re.compile(
    r"\b(it |information technology|system|infrastructure|operations|"
    r"ops|tech|digital|platform|enterprise|workplace tech|"
    r"internal tool|it manager|it director)\b"
)

def parse_function(title):
    if not isinstance(title, str):
        return "other"
    t = title.lower()
    if HR_PATTERN.search(t):
        return "hr_people"
    if OPS_IT_PATTERN.search(t):
        return "ops_it"
    return "other"

df["job_function"] = df["job_title"].apply(parse_function)
df["is_hr_function"]    = (df["job_function"] == "hr_people").astype(int)
df["is_ops_it_function"] = (df["job_function"] == "ops_it").astype(int)

# ── 3. Knowledge Industry — who has structural silos ─────────────────────────
# WHY: These industries have regulated knowledge, complex org structures,
#      distributed teams, or high cost of knowledge loss (turnover, compliance).
#      They feel the silo problem most acutely.
#
#      Finance/Insurance → compliance, advisor knowledge, branch silos
#      Government/Non-profit → program knowledge, onboarding churn
#      Consulting/Professional Services → project knowledge, mentoring
#      Education → faculty/staff development, knowledge retention
#      Real Estate/Property → local market knowledge silos
#      Manufacturing → shop-floor to office knowledge gaps

KNOWLEDGE_INDUSTRY = re.compile(
    r"\b(financ|bank|insurance|invest|capital|asset|accounting|audit|"
    r"government|public sector|non.profit|npo|ngo|association|policy|"
    r"consult|professional service|management consult|advisory|"
    r"educat|university|school|academ|e.learning|training|"
    r"real estate|property|construction|architecture|"
    r"manufactur|industrial|aerospace|automotive|"
    r"legal|law|compliance|regulatory|"
    r"healthcare|hospital|clinic|medic)\b"
)

SOCIAL_INDUSTRY = re.compile(
    r"\b(social media|entertainment|gaming|game|media|advertis|"
    r"marketing agency|creative agency|influencer|content creator)\b"
)

def classify_industry(ind):
    if not isinstance(ind, str):
        return "unknown"
    t = ind.lower()
    if KNOWLEDGE_INDUSTRY.search(t):
        return "knowledge_intensive"
    if SOCIAL_INDUSTRY.search(t):
        return "social_first"
    # Tech/SaaS are middle ground — they have silos but also have other tools
    if re.search(r"\b(software|saas|tech|it service|information technology|cloud|ai|data)\b", t):
        return "tech"
    return "other"

df["industry_type"] = df["industry"].apply(classify_industry)
df["is_knowledge_industry"] = (df["industry_type"] == "knowledge_intensive").astype(int)
df["is_tech_industry"]      = (df["industry_type"] == "tech").astype(int)

# ── 4. Silo Size — companies big enough to have silos ────────────────────────
# WHY: Under 100 people, everyone knows everyone — no silo problem.
#      Over 5000, they already have L&D teams and enterprise tools.
#      The sweet spot: 100–1500 employees where silos form but aren't yet
#      solved by a dedicated team.

def parse_size(numeric_val, range_val):
    try:
        v = float(str(numeric_val).replace(",", ""))
        if 0 < v < 1_000_000:
            return v
    except (ValueError, TypeError):
        pass
    RANGE_MAP = {
        "1-50": 25, "51-200": 125, "201-500": 350, "500-1000": 750,
        "1000-5000": 2500, "5000-10000": 7500, "10000+": 15000,
        "1-200": 100, "1k+": 2500, "11-50": 30, "1001-5000": 2500,
    }
    if isinstance(range_val, str):
        key = range_val.strip().lower()
        if key in RANGE_MAP:
            return float(RANGE_MAP[key])
        m = re.search(r"(\d[\d,]*)", range_val)
        if m:
            return float(m.group(1).replace(",", ""))
    return None

df["company_size"] = df.apply(
    lambda r: parse_size(r["company_size_numeric"], r["company_size_range"]), axis=1
)

df["is_silo_size"] = df["company_size"].apply(
    lambda s: 1 if (s is not None and not pd.isna(s) and 100 <= s <= 1500) else 0
)

# ── 5. Buyer signal — Director or above + HR/Ops function ────────────────────
# WHY: The ideal contact is someone who owns the problem AND can approve the budget.
#      Director of HR or Head of People = perfect. VP = also good.

df["is_buyer_persona"] = (
    (df["seniority_label"].isin(["director", "vp", "c_suite"])) &
    (df["job_function"].isin(["hr_people", "ops_it"]))
).astype(int)

# ── 6. Engagement recency ─────────────────────────────────────────────────────
# WHY: Not a firmographic signal, but recency of last contact tells us
#      if the lead is still warm. We keep it as a soft signal.

from datetime import datetime as dt
REF = dt(2026, 4, 22)
def days_since(v):
    if not isinstance(v, str): return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try: return (REF - dt.strptime(v[:10], fmt)).days
        except: pass
    return None

df["days_since_contact"] = df["last_contact_date"].apply(days_since)

# ── Final feature set ─────────────────────────────────────────────────────────

FEATURES = [
    # Core problem-fit signals
    "is_knowledge_industry",   # industry has structural silo problem
    "is_hr_function",          # contact owns the problem
    "is_ops_it_function",      # contact deploys the tool
    "is_silo_size",            # company size creates silos (100-1500)
    "is_buyer_persona",        # director+ in HR/IT function
    "is_tech_industry",        # tech companies — moderate signal, often over-tooled
    # Continuous signals
    "seniority_score",         # 0-5 scale
    "company_size",            # raw headcount
    "days_since_contact",      # recency
]

TARGET = "converted"

print("\n=== FEATURE DISTRIBUTION: Paying vs Non-Paying ===")
print(f"{'Feature':<25} {'Paying':>10} {'Non-paying':>12} {'Lift':>8}  Interpretation")
print("-" * 85)

paying    = df[df[TARGET] == 1]
non_paying = df[df[TARGET] == 0]

interpretations = {
    "is_knowledge_industry": "Finance/Gov/Consulting/Education/Legal → structured silo problem",
    "is_hr_function":        "HR/People/Culture/Talent contact → owns the problem",
    "is_ops_it_function":    "IT/Ops contact → deploys the tool",
    "is_silo_size":          "100–1500 employees → big enough for silos",
    "is_buyer_persona":      "Director+ AND HR/IT function → problem owner + budget",
    "is_tech_industry":      "Tech/SaaS → moderate, often already over-tooled",
    "seniority_score":       "0-5 seniority scale (5=CEO, 3=Director, 1=IC)",
    "company_size":          "Raw headcount",
    "days_since_contact":    "Days since last contact (lower = more recent)",
}

for feat in FEATURES:
    if feat not in df.columns:
        df[feat] = 0
    p_mean = paying[feat].mean() if feat in paying.columns else 0
    n_mean = non_paying[feat].mean() if feat in non_paying.columns else 0
    lift = p_mean - n_mean
    interp = interpretations.get(feat, "")
    print(f"  {feat:<25} {p_mean:>10.3f} {n_mean:>12.3f} {lift:>+8.3f}  {interp}")

# ─────────────────────────────────────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────────────────────────────────────

X_raw = df[FEATURES].copy()
for col in X_raw.columns:
    if X_raw[col].dtype == bool:
        X_raw[col] = X_raw[col].astype(int)

y = df[TARGET].astype(int)

imputer = SimpleImputer(strategy="median")
X       = imputer.fit_transform(X_raw)
scaler  = StandardScaler()
X_sc    = scaler.fit_transform(X)

# SMOTE
k = min(3, int(y.sum()) - 1)
if k >= 1 and y.mean() < 0.3:
    smote = SMOTE(random_state=42, k_neighbors=k)
    X_sm,    y_sm = smote.fit_resample(X,    y)
    X_sc_sm, _    = smote.fit_resample(X_sc, y)
    print(f"\nSMOTE: {y.sum()} → {y_sm.sum()} positives")
else:
    X_sm, y_sm, X_sc_sm = X, y, X_sc

X_tr,    X_te,    y_tr, y_te = train_test_split(X_sm,    y_sm, test_size=0.2, stratify=y_sm, random_state=42)
X_sc_tr, X_sc_te             = X_sc_sm[:len(X_tr)], X_sc_sm[len(X_tr):]

MODELS = {
    "Logistic Regression": {
        "m": LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5, random_state=42),
        "Xtr": X_sc_tr, "Xte": X_sc_te, "Xall": X_sc,
        "why": (
            "Assigns a +/- weight to each feature. The most transparent model — "
            "reads like a scorecard. 'Being in a knowledge industry adds +0.4 to score.' "
            "Weak on non-linear combos but tells us the direction of every signal."
        ),
    },
    "Random Forest": {
        "m": RandomForestClassifier(n_estimators=300, max_depth=5, class_weight="balanced", random_state=42, n_jobs=-1),
        "Xtr": X_tr, "Xte": X_te, "Xall": X,
        "why": (
            "Builds 300 decision trees and votes. Captures combinations like "
            "'Director AND knowledge industry AND 200-1000 employees' as a single rule. "
            "Better than logistic regression when the ICP is a profile, not just a checklist."
        ),
    },
    "XGBoost": {
        "m": xgb_mod.XGBClassifier(
            n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=(len(y_sm)-y_sm.sum())/max(y_sm.sum(),1),
            use_label_encoder=False, eval_metric="logloss", random_state=42, verbosity=0),
        "Xtr": X_tr, "Xte": X_te, "Xall": X,
        "why": (
            "Gradient boosting — each new tree corrects the previous one's mistakes. "
            "Best performer on small, imbalanced tabular datasets like this. "
            "Handles missing data natively. Our production recommendation."
        ),
    },
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
trained = {}

print("\n=== MODEL RESULTS ===")
for name, cfg in MODELS.items():
    cv_X = X_sc_sm if cfg["Xtr"] is X_sc_tr else X_sm
    cv_s = cross_val_score(cfg["m"], cv_X, y_sm, cv=cv, scoring="roc_auc", n_jobs=-1)
    cfg["m"].fit(cfg["Xtr"], y_tr)
    proba = cfg["m"].predict_proba(cfg["Xte"])[:, 1]
    pred  = (proba >= 0.5).astype(int)
    roc   = roc_auc_score(y_te, proba) if len(np.unique(y_te)) > 1 else 0.0
    pr    = average_precision_score(y_te, proba) if len(np.unique(y_te)) > 1 else 0.0
    f1    = f1_score(y_te, pred, zero_division=0)
    results[name] = {"roc": roc, "pr": pr, "f1": f1, "cv_mean": cv_s.mean(), "cv_std": cv_s.std()}
    trained[name] = cfg["m"]
    print(f"\n  {name}")
    print(f"    CV AUC : {cv_s.mean():.3f} ± {cv_s.std():.3f}")
    print(f"    ROC-AUC: {roc:.3f}   PR-AUC: {pr:.3f}   F1: {f1:.3f}")
    print(f"    WHY: {cfg['why'][:90]}...")

best_name = max(results, key=lambda k: results[k]["pr"])
best_m    = trained[best_name]
best_cfg  = MODELS[best_name]
print(f"\nBest model: {best_name} (PR-AUC = {results[best_name]['pr']:.3f})")

# ── Logistic Regression weights (scorecard) ───────────────────────────────────
lr = trained["Logistic Regression"]
lr_coef = pd.Series(lr.coef_[0], index=FEATURES).sort_values(ascending=False)
print("\n=== LOGISTIC REGRESSION SCORECARD ===")
print("  (What the model learned — each feature's contribution)")
for feat, coef in lr_coef.items():
    bar = "+" * int(abs(coef) * 10) if coef > 0 else "-" * int(abs(coef) * 10)
    print(f"  {feat:<25} {coef:+.3f}  {bar}")

# ── SHAP ──────────────────────────────────────────────────────────────────────
print("\nComputing SHAP...")
try:
    explainer = shap.TreeExplainer(best_m)
    sv = explainer.shap_values(best_cfg["Xte"])
    if isinstance(sv, list): sv = sv[1]
    sv = np.array(sv)
    if sv.ndim == 3: sv = sv[:, :, 1]
    mean_shap = np.abs(sv).mean(axis=0)
    top_idx   = np.argsort(mean_shap)[::-1]
    top_shap  = [(FEATURES[i], mean_shap[i]) for i in top_idx]
    print("  SHAP computed OK")
except Exception as e:
    print(f"  SHAP failed: {e}")
    top_shap = []

# ── Score all leads ───────────────────────────────────────────────────────────
X_all_imp = imputer.transform(X_raw)
X_all_sc  = scaler.transform(X_all_imp)
X_score   = X_all_sc if best_cfg["Xall"] is X_sc else X_all_imp

proba_all = best_m.predict_proba(X_score)[:, 1]
df["icp_v2_score"] = (proba_all * 100).round(1)

def tier(s):
    if s >= 75: return "Strong ICP"
    if s >= 55: return "Likely ICP"
    if s >= 35: return "Weak ICP"
    return "Not ICP"

df["icp_v2_tier"] = df["icp_v2_score"].apply(tier)

# ─────────────────────────────────────────────────────────────────────────────
# CHARTS
# ─────────────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 20))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.5, wspace=0.4)

# ── Chart A: Feature lift — paying vs non-paying ──────────────────────────────
ax_a = fig.add_subplot(gs[0, :])
binary_feats = ["is_knowledge_industry", "is_hr_function", "is_ops_it_function",
                "is_silo_size", "is_buyer_persona", "is_tech_industry"]
labels_a = ["Knowledge\nIndustry", "HR / People\nFunction", "IT / Ops\nFunction",
            "Silo Size\n100-1500 emp", "Buyer Persona\n(Dir+ & HR/IT)", "Tech\nIndustry"]
pay_vals  = [paying[f].mean() for f in binary_feats]
non_vals  = [non_paying[f].mean() for f in binary_feats]

x = np.arange(len(binary_feats))
w = 0.35
ax_a.bar(x - w/2, pay_vals, w, label="Paying customers", color="#2ecc71", alpha=0.88, edgecolor="white")
ax_a.bar(x + w/2, non_vals, w, label="Non-paying leads",  color="#e74c3c", alpha=0.88, edgecolor="white")
ax_a.set_xticks(x); ax_a.set_xticklabels(labels_a, fontsize=10)
ax_a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(xmax=1))
ax_a.set_title("ICP v2: What separates paying customers from non-paying leads\n(problem-driven features only)", fontsize=13, fontweight="bold")
ax_a.legend(fontsize=10); ax_a.set_ylim(0, 0.75); ax_a.grid(axis="y", alpha=0.3)
for bars in [ax_a.patches[:len(x)], ax_a.patches[len(x):]]:
    for bar in bars:
        h = bar.get_height()
        if h > 0.01:
            ax_a.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.0%}",
                      ha="center", va="bottom", fontsize=9, fontweight="bold")

# ── Chart B: SHAP feature importance ─────────────────────────────────────────
ax_b = fig.add_subplot(gs[1, 0])
if top_shap:
    feats_shap = [f for f, _ in top_shap]
    vals_shap  = [v for _, v in top_shap]
    colors_b = ["#2ecc71" if lr_coef.get(f, 0) >= 0 else "#e74c3c" for f in feats_shap]
    ax_b.barh(feats_shap[::-1], vals_shap[::-1], color=colors_b[::-1], alpha=0.85, edgecolor="white")
    ax_b.set_xlabel("Mean |SHAP value|")
    ax_b.set_title(f"Feature Importance (SHAP)\n{best_name}", fontsize=11, fontweight="bold")
    ax_b.grid(axis="x", alpha=0.3)
    green_patch = matplotlib.patches.Patch(color="#2ecc71", alpha=0.85, label="Boosts ICP score")
    red_patch   = matplotlib.patches.Patch(color="#e74c3c", alpha=0.85, label="Lowers ICP score")
    ax_b.legend(handles=[green_patch, red_patch], fontsize=8)

# ── Chart C: Logistic Regression scorecard ───────────────────────────────────
ax_c = fig.add_subplot(gs[1, 1])
feat_labels = {
    "is_buyer_persona":       "Buyer Persona\n(Dir+ & HR/IT)",
    "is_knowledge_industry":  "Knowledge Industry",
    "is_hr_function":         "HR / People Function",
    "is_silo_size":           "Silo Size (100-1500)",
    "is_ops_it_function":     "IT / Ops Function",
    "seniority_score":        "Seniority Score",
    "company_size":           "Company Size",
    "is_tech_industry":       "Tech Industry",
    "days_since_contact":     "Days Since Contact",
}
coefs = [(feat_labels.get(f, f), lr_coef[f]) for f in FEATURES if f in lr_coef.index]
coefs_sorted = sorted(coefs, key=lambda x: x[1], reverse=True)
labels_c = [c[0] for c in coefs_sorted]
vals_c   = [c[1] for c in coefs_sorted]
colors_c = ["#2ecc71" if v >= 0 else "#e74c3c" for v in vals_c]
ax_c.barh(labels_c[::-1], vals_c[::-1], color=colors_c[::-1], alpha=0.85, edgecolor="white")
ax_c.axvline(0, color="black", linewidth=0.8)
ax_c.set_xlabel("Logistic Regression Coefficient (positive = boosts ICP score)")
ax_c.set_title("Scorecard Weights\n(Logistic Regression — most interpretable)", fontsize=11, fontweight="bold")
ax_c.grid(axis="x", alpha=0.3)

# ── Chart D: Model comparison ─────────────────────────────────────────────────
ax_d = fig.add_subplot(gs[2, 0])
model_names = list(results.keys())
roc_vals = [results[n]["roc"] for n in model_names]
pr_vals  = [results[n]["pr"]  for n in model_names]
f1_vals  = [results[n]["f1"]  for n in model_names]
x_d = np.arange(len(model_names))
w_d = 0.25
ax_d.bar(x_d - w_d, roc_vals, w_d, label="ROC-AUC", color="#4C72B0", alpha=0.85)
ax_d.bar(x_d,       pr_vals,  w_d, label="PR-AUC",  color="#DD8452", alpha=0.85)
ax_d.bar(x_d + w_d, f1_vals,  w_d, label="F1",      color="#55A868", alpha=0.85)
ax_d.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Random baseline")
ax_d.set_xticks(x_d); ax_d.set_xticklabels([n.replace(" ", "\n") for n in model_names], fontsize=9)
ax_d.set_ylim(0, 1.1); ax_d.set_title("Model Performance Comparison\n(ICP v2 — problem-driven features)", fontsize=11, fontweight="bold")
ax_d.legend(fontsize=8); ax_d.grid(axis="y", alpha=0.3)
for bar in ax_d.patches:
    h = bar.get_height()
    if h > 0.5:
        ax_d.text(bar.get_x() + bar.get_width()/2, h + 0.01, f"{h:.2f}",
                  ha="center", va="bottom", fontsize=8)

# ── Chart E: ICP tier distribution ───────────────────────────────────────────
ax_e = fig.add_subplot(gs[2, 1])
tier_order  = ["Strong ICP", "Likely ICP", "Weak ICP", "Not ICP"]
tier_colors = ["#27ae60", "#2ecc71", "#f39c12", "#e74c3c"]
tier_counts = df["icp_v2_tier"].value_counts()
counts_e = [tier_counts.get(t, 0) for t in tier_order]
bars_e = ax_e.barh(tier_order[::-1], counts_e[::-1], color=tier_colors[::-1],
                   alpha=0.87, edgecolor="white", height=0.5)
for bar, count in zip(bars_e, counts_e[::-1]):
    ax_e.text(bar.get_width() + 3, bar.get_y() + bar.get_height()/2,
              f"{count} leads", va="center", fontsize=11, fontweight="bold")
ax_e.set_xlabel("Number of leads"); ax_e.set_xlim(0, max(counts_e) * 1.25)
ax_e.set_title("ICP v2 Score Distribution\n(575 total leads)", fontsize=11, fontweight="bold")
ax_e.grid(axis="x", alpha=0.3)

plt.savefig("icp_v2_dashboard.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nDashboard saved: icp_v2_dashboard.png")

# ── Print top leads to approach ───────────────────────────────────────────────
print("\n=== TOP NON-PAYING LEADS TO APPROACH ===")
candidates = df[
    (df["converted"] == 0) &
    (df["icp_v2_score"] >= 40) &
    (df["outcome_class"].isin(["in_progress", "inactive", "canceled"]))
].copy()

show_cols = ["company", "job_title", "industry", "company_size_range",
             "outcome_class", "icp_v2_score", "icp_v2_tier"]
print(candidates[show_cols].sort_values("icp_v2_score", ascending=False)
      .head(25).to_string(index=False))

# ── Write report ──────────────────────────────────────────────────────────────
tier_labels_desc = {
    "Strong ICP": "≥75 — approach proactively, personalized outreach",
    "Likely ICP":  "55–74 — add to drip, follow up quickly if they engage",
    "Weak ICP":    "35–54 — newsletter only unless they inbound",
    "Not ICP":     "<35 — do not pursue proactively",
}

with open(OUT_REPORT, "w") as f:
    f.write(f"# ICP Model v2 — LEAD.bot\n")
    f.write(f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
    f.write("## Design Philosophy\n\n")
    f.write("This model is built only from LEAD.bot's own customer data, interpreted through\n")
    f.write("the lens of the problem we solve: **breaking knowledge silos** in mid-size organizations.\n\n")
    f.write("We deliberately excluded:\n")
    f.write("- Platform (Slack/Teams): reflects our sales reach, not the problem fit\n")
    f.write("- Region: reflects our marketing, not the problem fit\n")
    f.write("- Competitor customer data: assumption-driven, not data-driven\n\n")
    f.write("We focus on:\n")
    f.write("- Does the industry have a structural knowledge silo problem?\n")
    f.write("- Is the contact the person who owns that problem?\n")
    f.write("- Is the company the right size (big enough for silos, small enough to act)?\n")
    f.write("- Does the contact have the seniority to buy?\n\n")

    f.write("## Features & Why We Used Them\n\n")
    f.write("| Feature | What it captures | Why it predicts ICP fit |\n")
    f.write("|---|---|---|\n")
    feature_explanations = [
        ("is_knowledge_industry", "Finance, Gov, Consulting, Education, Legal, Manufacturing, Healthcare",
         "These industries have compliance requirements, distributed teams, or high cost of knowledge loss. Silos cost them money."),
        ("is_hr_function",        "HR, People Ops, Culture, Talent, L&D in the job title",
         "This person owns the knowledge-sharing problem. They've already identified it and are looking for a solution."),
        ("is_ops_it_function",    "IT, Operations, Systems, Infrastructure in the job title",
         "Second buyer persona — they implement and manage the tool, often championing it to HR."),
        ("is_silo_size",          "Company headcount 100–1500",
         "Under 100: everyone knows each other, no silo. Over 1500: they already have L&D teams. The pain peak is in between."),
        ("is_buyer_persona",      "Director+ AND (HR or IT function)",
         "The intersection of problem ownership and budget authority. This is your target contact."),
        ("seniority_score",       "0–5 numerical seniority",
         "Director=3, VP=4, C-suite=5. Higher = more likely to recognize and fund a solution."),
        ("company_size",          "Raw headcount",
         "Continuous version of silo_size. Pays off in the 200–1000 range."),
        ("is_tech_industry",      "Software, SaaS, AI, Cloud companies",
         "Moderate signal. Tech companies have silos too but are often over-tooled (Notion, Confluence, Slack). Neutral-to-weak ICP."),
        ("days_since_contact",    "Days since last contact",
         "Recency. A lead contacted 30 days ago is warmer than one from 2 years ago."),
    ]
    for feat, what, why in feature_explanations:
        f.write(f"| `{feat}` | {what} | {why} |\n")

    f.write("\n## Models Used\n\n")
    for name, cfg in MODELS.items():
        r = results[name]
        f.write(f"### {name}\n")
        f.write(f"**Why we used it:** {cfg['why']}\n\n")
        f.write(f"- CV AUC: {r['cv_mean']:.3f} ± {r['cv_std']:.3f}\n")
        f.write(f"- ROC-AUC: {r['roc']:.3f} | PR-AUC: {r['pr']:.3f} | F1: {r['f1']:.3f}\n\n")

    f.write(f"**Winner: {best_name}** — used for scoring\n\n")

    f.write("## Logistic Regression Scorecard\n\n")
    f.write("Most transparent model — reads like a sales qualification scorecard:\n\n")
    f.write("| Feature | Weight | Effect |\n|---|---|---|\n")
    for feat, coef in lr_coef.sort_values(ascending=False).items():
        effect = "↑ Boosts ICP score" if coef > 0 else "↓ Reduces ICP score"
        f.write(f"| {feat} | {coef:+.3f} | {effect} |\n")

    f.write("\n## ICP Tier Distribution\n\n")
    f.write("| Tier | Count | Threshold & Action |\n|---|---|---|\n")
    for tier_name in tier_order:
        count = tier_counts.get(tier_name, 0)
        f.write(f"| {tier_name} | {count} | {tier_labels_desc[tier_name]} |\n")

    f.write("\n## Top Leads to Approach\n\n")
    f.write("Non-paying leads with ICP v2 score ≥ 40:\n\n")
    f.write("| Company | Job Title | Industry | Size | Status | Score | Tier |\n")
    f.write("|---|---|---|---|---|---|---|\n")
    for _, row in candidates[show_cols].sort_values("icp_v2_score", ascending=False).head(25).iterrows():
        f.write(f"| {row.get('company','')} | {row.get('job_title','')} | "
                f"{row.get('industry','')} | {row.get('company_size_range','')} | "
                f"{row.get('outcome_class','')} | {row['icp_v2_score']} | {row['icp_v2_tier']} |\n")

    f.write("\n## Key Findings\n\n")
    f.write("### What we learned from YOUR data\n\n")
    for feat in binary_feats:
        p = paying[feat].mean() if feat in paying.columns else 0
        n = non_paying[feat].mean() if feat in non_paying.columns else 0
        lift = p - n
        arrow = "↑" if lift > 0 else "↓"
        f.write(f"- **{feat}**: {p:.0%} of paying customers vs {n:.0%} of non-paying ({arrow} {abs(lift):.0%} lift)\n")

    f.write("\n### The honest caveats\n\n")
    f.write("- 18 paying customers is a small training set. The model will improve significantly as you add more.\n")
    f.write("- Platform (Slack/Teams) and region were excluded intentionally — they reflect YOUR reach, not the problem.\n")
    f.write("  Once you expand to Teams or new regions, re-add them as features.\n")
    f.write("- This model predicts ICP fit based on current messaging and product. As positioning evolves, retrain.\n")

print(f"\nReport: {OUT_REPORT}")
print(f"Scores: {OUT_SCORES}")
df[["company","job_title","industry","company_size_range","outcome_class",
    "job_function","seniority_label","industry_type",
    "icp_v2_score","icp_v2_tier"]].sort_values(
    "icp_v2_score", ascending=False).to_csv(OUT_SCORES, index=False)
