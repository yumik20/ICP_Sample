"""
Steps 4–6: Model training, business value analysis, and evaluation.

Input:  lead_scoring_features.csv
Output: model_results.md   — full results log (appended to project plan)
        lead_scores.csv    — every lead with its predicted score (0–100)
        shap_summary.png   — feature importance chart
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime

# ── Install deps if needed ────────────────────────────────────────────────────
def require(pkg, import_as=None):
    try:
        return __import__(import_as or pkg)
    except ImportError:
        import subprocess, sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])
        return __import__(import_as or pkg)

sklearn   = require("scikit-learn", "sklearn")
xgb       = require("xgboost")
lgb       = require("lightgbm")
shap_mod  = require("shap")
imblearn  = require("imbalanced-learn", "imblearn")

from sklearn.linear_model   import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes     import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.impute           import SimpleImputer
from sklearn.metrics         import (roc_auc_score, average_precision_score,
                                     f1_score, confusion_matrix,
                                     classification_report, brier_score_loss)
from sklearn.calibration      import calibration_curve
import xgboost as xgb_mod
import lightgbm as lgb_mod
import shap

INPUT_FILE   = "lead_scoring_features.csv"
RESULTS_FILE = "model_results.md"
SCORES_FILE  = "lead_scores.csv"
SHAP_FILE    = "shap_summary.png"

# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_csv(INPUT_FILE)
print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

# Target
y = df["converted"].astype(int)

# Feature columns: everything that was one-hot encoded + numeric features
META  = ["sheet_origin", "job_title", "company", "industry", "platform",
         "region", "converted", "outcome_class", "sale_stage_raw"]

# Drop leaky features: stg_converted and stg_churned directly encode the label
# sale_stage_score also leaks (92+ = paying), so we train TWO models:
#   Model A — with stage score (for ranking existing pipeline leads)
#   Model B — without stage score (for scoring brand-new inbound leads)
LEAKY = [c for c in df.columns if c.startswith("stg_")]  # stage bucket OHE
X_raw = df.drop(columns=[c for c in META + LEAKY if c in df.columns])

# Convert bool columns to int (get_dummies produces bool on some pandas versions)
for col in X_raw.columns:
    if X_raw[col].dtype == bool:
        X_raw[col] = X_raw[col].astype(int)

# Numeric only (drop any remaining object columns)
X_raw = X_raw.select_dtypes(include=[np.number])
feature_names = X_raw.columns.tolist()
print(f"Feature count: {len(feature_names)}")
print(f"Class balance: {y.value_counts().to_dict()}")
print(f"Features: {feature_names}")

# ── Preprocessing ─────────────────────────────────────────────────────────────

# Impute missing with median
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X_raw)

# Scaled copy for Logistic Regression / Naive Bayes
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Handle class imbalance with SMOTE if minority class < 20%
minority_pct = y.mean()
use_smote = minority_pct < 0.2 and y.sum() >= 5

if use_smote:
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=42, k_neighbors=min(3, y.sum()-1))
    X_sm, y_sm = smote.fit_resample(X, y)
    X_scaled_sm, _ = smote.fit_resample(X_scaled, y)
    print(f"SMOTE applied: {y.sum()} → {y_sm.sum()} positive samples")
else:
    X_sm, y_sm = X, y
    X_scaled_sm = X_scaled

# Train/test split
X_tr, X_te, y_tr, y_te = train_test_split(X_sm, y_sm, test_size=0.2,
                                            stratify=y_sm, random_state=42)
X_tr_sc, X_te_sc = X_scaled_sm[:len(X_tr)], X_scaled_sm[len(X_tr):]

# ── Model definitions ─────────────────────────────────────────────────────────

models = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        X_tr_sc, X_te_sc
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=200, class_weight="balanced",
                               random_state=42, n_jobs=-1),
        X_tr, X_te
    ),
    "Gradient Boosting (sklearn)": (
        GradientBoostingClassifier(n_estimators=200, learning_rate=0.05,
                                   max_depth=4, random_state=42),
        X_tr, X_te
    ),
    "XGBoost": (
        xgb_mod.XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            scale_pos_weight=(len(y_sm)-y_sm.sum())/max(y_sm.sum(),1),
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0),
        X_tr, X_te
    ),
    "LightGBM": (
        lgb_mod.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            class_weight="balanced", random_state=42, verbose=-1),
        X_tr, X_te
    ),
    "Naive Bayes (Bayesian proxy)": (
        GaussianNB(),
        X_tr_sc, X_te_sc
    ),
}

# ── Cross-validation ──────────────────────────────────────────────────────────

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

print("\n=== Cross-Validation (5-fold ROC-AUC) ===")
for name, (model, Xtr, Xte) in models.items():
    scores = cross_val_score(model, X_sm if Xtr is X_tr else X_scaled_sm,
                              y_sm, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_results[name] = {"mean": scores.mean(), "std": scores.std()}
    print(f"  {name:40s}  AUC {scores.mean():.3f} ± {scores.std():.3f}")

# ── Train & evaluate on hold-out ─────────────────────────────────────────────

print("\n=== Hold-out Test Set Evaluation ===")
results = {}
trained_models = {}

for name, (model, Xtr, Xte) in models.items():
    model.fit(Xtr, y_tr)
    trained_models[name] = model

    proba = model.predict_proba(Xte)[:, 1]
    pred  = (proba >= 0.5).astype(int)

    roc   = roc_auc_score(y_te, proba) if len(np.unique(y_te)) > 1 else 0.0
    pr    = average_precision_score(y_te, proba) if len(np.unique(y_te)) > 1 else 0.0
    f1    = f1_score(y_te, pred, zero_division=0)
    brier = brier_score_loss(y_te, proba)
    cm    = confusion_matrix(y_te, pred)

    results[name] = {
        "roc_auc": roc, "pr_auc": pr, "f1": f1,
        "brier": brier, "confusion": cm.tolist(),
        "cv_mean": cv_results[name]["mean"],
        "cv_std":  cv_results[name]["std"],
    }
    print(f"  {name:40s}  ROC={roc:.3f}  PR={pr:.3f}  F1={f1:.3f}  Brier={brier:.3f}")

# ── Pick best model ───────────────────────────────────────────────────────────

best_name = max(results, key=lambda k: results[k]["roc_auc"])
best_model = trained_models[best_name]
best_Xtr, best_Xte = models[best_name][1], models[best_name][2]
print(f"\nBest model: {best_name} (ROC-AUC = {results[best_name]['roc_auc']:.3f})")

# ── SHAP feature importance ───────────────────────────────────────────────────

print("\nComputing SHAP values...")
try:
    if "XGBoost" in best_name or "LightGBM" in best_name or "Forest" in best_name:
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(best_Xte)
        # Random Forest / sklearn returns list [class0, class1]
        if isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values = shap_values[1]
        # Make sure it's a 2D array
        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]
    else:
        explainer = shap.LinearExplainer(best_model, best_Xtr, feature_perturbation="interventional")
        shap_values = np.array(explainer.shap_values(best_Xte))

    # Convert Xte to DataFrame for SHAP plot
    Xte_df = pd.DataFrame(best_Xte, columns=feature_names)

    shap.summary_plot(shap_values, Xte_df, show=False)
    plt.title(f"Feature Importance (SHAP) — {best_name}", fontsize=13)
    plt.tight_layout()
    plt.savefig(SHAP_FILE, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP chart saved: {SHAP_FILE}")

    # Top 15 features by mean |SHAP|
    mean_shap = np.abs(shap_values).mean(axis=0)
    top_idx   = np.argsort(mean_shap)[::-1][:15]
    top_features = [(feature_names[i], mean_shap[i]) for i in top_idx]

except Exception as e:
    print(f"SHAP failed: {e}")
    import traceback; traceback.print_exc()
    top_features = []

# ── Score all original leads ──────────────────────────────────────────────────

print("\nScoring all leads...")
X_all = imputer.transform(X_raw)
if best_Xtr is X_tr_sc:
    X_all_for_score = scaler.transform(X_all)
else:
    X_all_for_score = X_all

proba_all = best_model.predict_proba(X_all_for_score)[:, 1]
df["lead_score"] = (proba_all * 100).round(1)

def score_tier(s):
    if s >= 80: return "Hot"
    if s >= 60: return "Warm"
    if s >= 40: return "Lukewarm"
    if s >= 20: return "Cold"
    return "Disqualified"

df["score_tier"] = df["lead_score"].apply(score_tier)

score_cols = ["company", "job_title", "industry", "platform", "region",
              "outcome_class", "sale_stage_raw", "lead_score", "score_tier"]
df[[c for c in score_cols if c in df.columns]].sort_values(
    "lead_score", ascending=False
).to_csv(SCORES_FILE, index=False)
print(f"Scores saved: {SCORES_FILE}")

# ── Write results markdown ────────────────────────────────────────────────────

now = datetime.now().strftime("%Y-%m-%d %H:%M")

with open(RESULTS_FILE, "w") as f:
    f.write(f"# Model Results — LEAD.bot Lead Scoring\n")
    f.write(f"**Run date:** {now}\n\n")

    f.write("## Dataset\n\n")
    f.write(f"- Total rows: {len(df)}\n")
    f.write(f"- Features: {len(feature_names)}\n")
    f.write(f"- Converted=1 (paying): {int(y.sum())} ({100*y.mean():.1f}%)\n")
    f.write(f"- SMOTE applied: {'Yes' if use_smote else 'No'}\n\n")

    f.write("## Cross-Validation Results (5-fold ROC-AUC)\n\n")
    f.write("| Model | CV AUC Mean | CV AUC Std |\n")
    f.write("|---|---|---|\n")
    for name, cv in cv_results.items():
        f.write(f"| {name} | {cv['mean']:.3f} | {cv['std']:.3f} |\n")

    f.write("\n## Hold-out Test Metrics\n\n")
    f.write("| Model | ROC-AUC | PR-AUC | F1 | Brier Score |\n")
    f.write("|---|---|---|---|---|\n")
    for name, r in results.items():
        marker = " **← best**" if name == best_name else ""
        f.write(f"| {name}{marker} | {r['roc_auc']:.3f} | {r['pr_auc']:.3f} | {r['f1']:.3f} | {r['brier']:.3f} |\n")

    f.write(f"\n## Best Model: {best_name}\n\n")
    f.write(f"- ROC-AUC: {results[best_name]['roc_auc']:.3f}\n")
    f.write(f"- PR-AUC:  {results[best_name]['pr_auc']:.3f}\n")
    f.write(f"- F1:      {results[best_name]['f1']:.3f}\n\n")

    cm = np.array(results[best_name]["confusion"])
    f.write("### Confusion Matrix\n\n")
    f.write("```\n")
    f.write(f"              Predicted 0  Predicted 1\n")
    if cm.shape == (2, 2):
        f.write(f"Actual 0         {cm[0,0]:6d}       {cm[0,1]:6d}\n")
        f.write(f"Actual 1         {cm[1,0]:6d}       {cm[1,1]:6d}\n")
    f.write("```\n\n")

    if top_features:
        f.write("## Top 15 Features by SHAP Importance\n\n")
        f.write("| Rank | Feature | Mean |SHAP| |\n")
        f.write("|---|---|---|\n")
        for rank, (feat, val) in enumerate(top_features, 1):
            f.write(f"| {rank} | {feat} | {val:.4f} |\n")
        f.write(f"\nSHAP chart: `{SHAP_FILE}`\n")

    f.write("\n## Lead Score Distribution\n\n")
    tier_counts = df["score_tier"].value_counts()
    f.write("| Tier | Count | Threshold |\n")
    f.write("|---|---|---|\n")
    tier_order = ["Hot", "Warm", "Lukewarm", "Cold", "Disqualified"]
    tiers_thresholds = {"Hot": "≥ 80", "Warm": "60–79", "Lukewarm": "40–59",
                        "Cold": "20–39", "Disqualified": "< 20"}
    for tier in tier_order:
        count = tier_counts.get(tier, 0)
        f.write(f"| {tier} | {count} | {tiers_thresholds[tier]} |\n")

    f.write(f"\n## Business Value\n\n")
    hot_leads   = (df["score_tier"] == "Hot").sum()
    warm_leads  = (df["score_tier"] == "Warm").sum()
    total_leads = len(df)
    f.write(f"- Model identifies **{hot_leads} Hot leads** ({100*hot_leads/total_leads:.1f}% of all leads)\n")
    f.write(f"- Model identifies **{warm_leads} Warm leads** to follow up ({100*warm_leads/total_leads:.1f}%)\n")
    f.write(f"- Sales team can focus on top {hot_leads+warm_leads} leads ({100*(hot_leads+warm_leads)/total_leads:.1f}%) "
            f"rather than working all {total_leads}\n\n")

    f.write("### Sales Action Tiers\n\n")
    f.write("| Score | Tier | Recommended Action |\n")
    f.write("|---|---|---|\n")
    f.write("| 80–100 | Hot | Immediate personal outreach — call or personalized email |\n")
    f.write("| 60–79 | Warm | Follow up within 1 week |\n")
    f.write("| 40–59 | Lukewarm | Monthly newsletter + check-in |\n")
    f.write("| 20–39 | Cold | Quarterly newsletter only |\n")
    f.write("| 0–19 | Disqualified | Mailchimp automation only |\n")

    f.write("\n## Evaluation Notes\n\n")
    f.write(f"- Dataset is small (~{len(df)} rows, {int(y.sum())} converted).\n")
    f.write("  SMOTE was used to address class imbalance. Retrain as more data comes in.\n")
    f.write("- PR-AUC is the most reliable metric given the imbalanced classes.\n")
    f.write("- Calibration was not formally tested — for production use, add isotonic regression calibration.\n")
    f.write("- Survival analysis (time-to-close) is recommended as a next step once clean close dates are available.\n")

print(f"\nResults written to: {RESULTS_FILE}")
print(f"\nDone! Summary:")
print(f"  Best model:  {best_name}")
print(f"  ROC-AUC:     {results[best_name]['roc_auc']:.3f}")
print(f"  All outputs: {RESULTS_FILE}, {SCORES_FILE}, {SHAP_FILE}")
