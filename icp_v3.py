"""
ICP v3 — Positive-Unlabeled (PU) Similarity Scoring

The core fix: non-paying leads are NOT labeled as "bad ICP".
They are UNLABELED — we simply don't know yet.

Only paying customers are ground truth positives.
Everyone else is scored by how similar they are to that confirmed ICP cluster.

Approach:
  1. Build a profile of paying customers in feature space
  2. Score every lead by similarity to that centroid (Mahalanobis distance)
  3. Also train a PU-style model: bootstrap negatives from the LEAST similar
     unlabeled leads (those that are clearly different from payers)
  4. Output: icp_v3_scores.csv + charts + report
"""

import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

BG     = "#0f1117"
CARD   = "#1a1d2e"
TEXT   = "#e0e0e0"
GREEN  = "#22c55e"
PURPLE = "#6c63ff"
YELLOW = "#f59e0b"
RED    = "#ef4444"
GRAY   = "#4b5563"

ICP_FEATURES = [
    # Size signal — strongest
    "sz_mid",           # 200–1499 employees — sweet spot
    "sz_small",         # 50–199
    "sz_large",         # 1500–4999
    "sz_enterprise",    # 5000–9999
    "sz_mega",          # 10000+
    "sz_micro",         # <50
    "company_size",     # raw headcount

    # Industry — knowledge-intensive = strong signal
    "ind_finance",
    "ind_government",
    "ind_consulting",
    "ind_education",
    "ind_manufacturing",
    "ind_healthcare",
    "ind_media",
    "ind_realestate",
    "ind_tech",         # weak/negative signal
    "ind_other",

    # Seniority / function
    "sen_c_suite",
    "sen_director",
    "sen_vp",
    "sen_manager",
    "sen_ic",
    "seniority_score",

    # Engagement recency
    "days_since_last_contact",
]

# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_csv("lead_scoring_features.csv")

# Convert bool columns
for col in df.columns:
    if df[col].dtype == bool or df[col].dtype == object:
        if df[col].isin([True, False, "True", "False"]).all():
            df[col] = df[col].map({"True": 1, "False": 0, True: 1, False: 0})

# Use only features that exist
features = [f for f in ICP_FEATURES if f in df.columns]
X = df[features].fillna(0).astype(float)

# Labels: 1 = confirmed ICP (paying OR canceled)
# Rationale: canceled customers bought into the vision — churn is a product/timing
# problem, not an ICP problem. Their firmographic profile is valid positive signal.
# Only in_progress, inactive, and unknown are treated as unlabeled.
paying   = df["converted"] == 1
churned  = df["outcome_class"] == "canceled"
positives = paying | churned
unlabeled = ~positives

print(f"Confirmed ICP (paying):   {paying.sum()}")
print(f"Confirmed ICP (churned):  {churned.sum()}  ← counted as ICP, churn = product maturity issue")
print(f"Total confirmed ICP:      {positives.sum()}")
print(f"Unlabeled leads:          {unlabeled.sum()}")
print(f"Features:                 {len(features)}")

# ── Step 1: Similarity scoring (centroid distance) ────────────────────────────
# How close is each lead to the centroid of paying customers?

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ICP centroid in scaled space
icp_centroid = X_scaled[positives].mean(axis=0)

# Cosine similarity to centroid
def cosine_similarity_to_centroid(X_s, centroid):
    norms = np.linalg.norm(X_s, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    X_norm = X_s / norms
    c_norm = centroid / (np.linalg.norm(centroid) + 1e-10)
    return X_norm @ c_norm

cos_sim = cosine_similarity_to_centroid(X_scaled, icp_centroid)

# Euclidean distance (inverted to similarity)
euc_dist = np.linalg.norm(X_scaled - icp_centroid, axis=1)
euc_sim  = 1 / (1 + euc_dist)

# Blend: 60% cosine, 40% euclidean proximity
similarity_raw = 0.6 * cos_sim + 0.4 * euc_sim

# Normalize to 0–100
s_min, s_max = similarity_raw.min(), similarity_raw.max()
similarity_score = (similarity_raw - s_min) / (s_max - s_min) * 100
df["similarity_score"] = similarity_score.round(1)

# ── Step 2: PU Learning — use clearly-dissimilar leads as proxy negatives ─────
# Instead of treating ALL non-payers as negatives, we identify the bottom 20%
# most dissimilar unlabeled leads and use those as "almost certainly not ICP"
# proxy negatives. This is the Spy/Bagging PU learning technique.

PROXY_NEG_PERCENTILE = 20   # bottom 20% similarity = proxy negative

threshold = np.percentile(similarity_score[unlabeled], PROXY_NEG_PERCENTILE)
proxy_negative = unlabeled & (similarity_score <= threshold)

# Training set: confirmed positives + proxy negatives only
train_mask = positives | proxy_negative
X_train = X_scaled[train_mask]
y_train  = positives[train_mask].astype(int).values

print(f"\nPU Learning training set:")
print(f"  Positives (confirmed ICP): {positives[train_mask].sum()}")
print(f"  Proxy negatives (bottom {PROXY_NEG_PERCENTILE}% dissimilar): {proxy_negative.sum()}")

# Train Random Forest on PU training set
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                             max_depth=6, random_state=42)
rf.fit(X_train, y_train)

# Score ALL leads
pu_proba = rf.predict_proba(X_scaled)[:, 1]
pu_min, pu_max = pu_proba.min(), pu_proba.max()
pu_score = (pu_proba - pu_min) / (pu_max - pu_min) * 100
df["pu_score"] = pu_score.round(1)

# Also train Isolation Forest as unsupervised outlier from ICP cluster
iso = IsolationForest(n_estimators=200, contamination=0.3, random_state=42)
iso.fit(X_scaled[positives])   # fit on ICP positives only
iso_scores = iso.score_samples(X_scaled)
iso_norm = (iso_scores - iso_scores.min()) / (iso_scores.max() - iso_scores.min()) * 100
df["isolation_score"] = iso_norm.round(1)

# ── Step 3: Ensemble ICP score ────────────────────────────────────────────────
# Blend: 40% similarity, 40% PU model, 20% isolation forest

df["icp_v3_score"] = (
    0.40 * df["similarity_score"] +
    0.40 * df["pu_score"] +
    0.20 * df["isolation_score"]
).round(1)

# Tier labels
def tier(score):
    if score >= 75: return "Strong ICP"
    if score >= 55: return "Likely ICP"
    if score >= 35: return "Weak ICP"
    return "Not ICP"

df["icp_v3_tier"] = df["icp_v3_score"].apply(tier)

# How did the paying customers score?
payer_scores = df.loc[positives, "icp_v3_score"]
payer_scores = df.loc[paying,   "icp_v3_score"]
churn_scores = df.loc[churned,  "icp_v3_score"]
print(f"\nPaying customer scores:  mean={payer_scores.mean():.1f}  min={payer_scores.min():.1f}  max={payer_scores.max():.1f}")
print(f"Churned customer scores: mean={churn_scores.mean():.1f}  min={churn_scores.min():.1f}  max={churn_scores.max():.1f}")

# ── Step 4: Feature importance ────────────────────────────────────────────────

importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

# ── Step 5: PCA for cluster visualization ─────────────────────────────────────

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df["pca1"] = X_pca[:, 0]
df["pca2"] = X_pca[:, 1]

# ── Step 6: Rank all leads ────────────────────────────────────────────────────

# Top ICP candidates that aren't paying yet
candidates = df[~positives].sort_values("icp_v3_score", ascending=False)

print(f"\nTop 15 ICP candidates (non-paying):")
print(candidates[["company","industry","outcome_class","icp_v3_score","icp_v3_tier"]].head(15).to_string(index=False))

# Save scores
out_cols = ["company","job_title","industry","outcome_class",
            "similarity_score","pu_score","isolation_score","icp_v3_score","icp_v3_tier"]
df[out_cols].sort_values("icp_v3_score", ascending=False).to_csv("icp_v3_scores.csv", index=False)
print("\nSaved: icp_v3_scores.csv")

# ── Step 7: Charts ────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 16), facecolor=BG)
gs  = GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

def style_ax(ax, title=""):
    ax.set_facecolor(CARD)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(colors=TEXT, labelsize=8)
    if title: ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=8)
    ax.xaxis.grid(True, color="#2a2d3e", zorder=0)
    ax.set_axisbelow(True)

# 1. Score distribution comparison: payers vs unlabeled
ax1 = fig.add_subplot(gs[0, :2])
style_ax(ax1, "ICP v3 Score Distribution — Paying Customers vs Unlabeled Leads")
bins = np.linspace(0, 100, 25)
ax1.hist(df.loc[~positives, "icp_v3_score"], bins=bins, color=GRAY, alpha=0.7, label="Unlabeled leads (in-progress / inactive)", zorder=3)
ax1.hist(df.loc[churned,    "icp_v3_score"], bins=bins, color=YELLOW, alpha=0.85, label="Churned (counted as ICP)", zorder=4)
ax1.hist(df.loc[paying,     "icp_v3_score"], bins=bins, color=GREEN, alpha=0.9, label="Paying customers ✓", zorder=5)
ax1.axvline(75, color=GREEN, linewidth=1, linestyle="--", alpha=0.6, label="Strong ICP threshold (75)")
ax1.axvline(55, color=PURPLE, linewidth=1, linestyle="--", alpha=0.6, label="Likely ICP threshold (55)")
ax1.set_xlabel("ICP v3 Score", color=TEXT, fontsize=9)
ax1.set_ylabel("Count", color=TEXT, fontsize=9)
ax1.legend(fontsize=8, facecolor=CARD, labelcolor=TEXT, framealpha=0.8)

# 2. Tier breakdown pie
ax2 = fig.add_subplot(gs[0, 2])
ax2.set_facecolor(CARD)
for sp in ax2.spines.values(): sp.set_visible(False)
tier_counts = df["icp_v3_tier"].value_counts().reindex(["Strong ICP","Likely ICP","Weak ICP","Not ICP"]).fillna(0)
colors_pie = [GREEN, PURPLE, YELLOW, RED]
wedges, _, autotexts = ax2.pie(tier_counts, labels=None, colors=colors_pie, autopct="%1.0f%%",
    startangle=90, pctdistance=0.72, wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2))
for at in autotexts: at.set(color=BG, fontsize=8, fontweight="bold")
ax2.legend(wedges, [f"{t} ({int(n)})" for t, n in zip(tier_counts.index, tier_counts.values)],
           loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=2,
           fontsize=7.5, facecolor=CARD, labelcolor=TEXT, framealpha=0.8)
ax2.set_title("Lead Tier Breakdown (PU Model)", color=TEXT, fontsize=10, fontweight="bold", pad=8)

# 3. PCA scatter — ICP landscape
ax3 = fig.add_subplot(gs[1, :2])
style_ax(ax3, "ICP Landscape — PCA Projection (colored by ICP v3 Score)")
sc = ax3.scatter(df.loc[~positives, "pca1"], df.loc[~positives, "pca2"],
                 c=df.loc[~positives, "icp_v3_score"], cmap="RdYlGn",
                 s=25, alpha=0.6, vmin=0, vmax=100, zorder=3, label="Unlabeled")
ax3.scatter(df.loc[churned, "pca1"], df.loc[churned, "pca2"],
            c=YELLOW, s=80, marker="D", zorder=4, label="Churned (ICP ✓)", edgecolors="white", linewidths=0.4, alpha=0.85)
ax3.scatter(df.loc[paying, "pca1"], df.loc[paying, "pca2"],
            c=GREEN, s=120, marker="*", zorder=5, label="Paying customer ✓", edgecolors="white", linewidths=0.5)
plt.colorbar(sc, ax=ax3, label="ICP v3 Score", fraction=0.04)
ax3.set_xlabel("PCA Component 1", color=TEXT, fontsize=8)
ax3.set_ylabel("PCA Component 2", color=TEXT, fontsize=8)
ax3.legend(fontsize=8, facecolor=CARD, labelcolor=TEXT, framealpha=0.8)
ax3.xaxis.grid(True, color="#2a2d3e")
ax3.yaxis.grid(True, color="#2a2d3e")

# 4. Feature importance
ax4 = fig.add_subplot(gs[1, 2])
style_ax(ax4, "Feature Importance (PU Random Forest)")
top_features = importances.head(12)
colors_f = [GREEN if v > importances.median() else PURPLE for v in top_features.values]
ax4.barh(top_features.index[::-1], top_features.values[::-1], color=colors_f[::-1], height=0.6, zorder=3)
ax4.set_xlabel("Importance", color=TEXT, fontsize=8)
ax4.tick_params(axis='y', labelsize=7.5)

# 5. Score vs outcome class — the key validation
ax5 = fig.add_subplot(gs[2, :2])
style_ax(ax5, "ICP v3 Score by Outcome Class — Are We Scoring Right?")
outcome_order = ["paying", "in_progress", "inactive", "canceled", "unknown"]
outcome_colors = [GREEN, PURPLE, GRAY, RED, YELLOW]
positions = []
labels_used = []
for i, (oc, col) in enumerate(zip(outcome_order, outcome_colors)):
    subset = df[df["outcome_class"] == oc]["icp_v3_score"]
    if len(subset) == 0: continue
    bp = ax5.boxplot(subset, positions=[i], widths=0.5, patch_artist=True,
                     medianprops=dict(color="white", linewidth=2),
                     boxprops=dict(facecolor=col, alpha=0.7),
                     whiskerprops=dict(color=TEXT), capprops=dict(color=TEXT),
                     flierprops=dict(marker="o", color=col, alpha=0.4, markersize=3))
    positions.append(i)
    labels_used.append(f"{oc}\n(n={len(subset)})")
ax5.set_xticks(positions)
ax5.set_xticklabels(labels_used, color=TEXT, fontsize=8)
ax5.set_ylabel("ICP v3 Score", color=TEXT, fontsize=8)
ax5.axhline(75, color=GREEN, linewidth=1, linestyle="--", alpha=0.5)
ax5.axhline(55, color=PURPLE, linewidth=1, linestyle="--", alpha=0.5)
ax5.set_ylim(0, 105)
ax5.xaxis.grid(False)

# 6. Top candidates table
ax6 = fig.add_subplot(gs[2, 2])
ax6.set_facecolor(CARD)
for sp in ax6.spines.values(): sp.set_visible(False)
ax6.axis("off")
top15 = candidates.head(10)
table_data = []
for _, r in top15.iterrows():
    ind = r.get("industry","")[:14] if pd.notna(r.get("industry","")) else ""
    oc  = r.get("outcome_class","")
    sc  = f"{r['icp_v3_score']:.0f}"
    table_data.append([ind, oc[:10], sc])
tbl = ax6.table(cellText=table_data,
                colLabels=["Industry", "Status", "Score"],
                bbox=[0, 0, 1, 1], cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
for (ri, ci), cell in tbl.get_celld().items():
    cell.set_edgecolor("#2a2d3e")
    if ri == 0:
        cell.set_facecolor(PURPLE)
        cell.set_text_props(color="white", fontweight="bold")
    else:
        score_val = float(table_data[ri-1][2]) if ri <= len(table_data) else 0
        cell.set_facecolor("#242738" if ri % 2 == 0 else CARD)
        color = GREEN if score_val >= 75 else (PURPLE if score_val >= 55 else TEXT)
        cell.set_text_props(color=color if ci == 2 else TEXT)
ax6.set_title("Top ICP Candidates (Non-Paying)", color=TEXT, fontsize=9, fontweight="bold", pad=8)

fig.suptitle("LEAD.bot ICP v3 — Positive-Unlabeled Similarity Scoring\n"
             "Paying + Churned = confirmed ICP. In-progress / inactive = unlabeled. Churn ≠ bad ICP.",
             color=TEXT, fontsize=13, fontweight="bold", y=0.98)

plt.savefig("icp_v3_chart.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("Saved: icp_v3_chart.png")

# ── Step 8: Summary report ────────────────────────────────────────────────────

tier_counts_full = df["icp_v3_tier"].value_counts()
paying_tier = df[positives]["icp_v3_tier"].value_counts()

print("\n" + "="*60)
print("ICP v3 SUMMARY")
print("="*60)
print(f"\nModel: Positive-Unlabeled (PU) + Cosine Similarity + Isolation Forest")
print(f"ICP positives: paying customers + churned customers (churn = product maturity, not ICP mismatch)")
print(f"Unlabeled: in-progress + inactive + unknown")
print()
print("All leads by tier:")
for t in ["Strong ICP","Likely ICP","Weak ICP","Not ICP"]:
    n = tier_counts_full.get(t, 0)
    print(f"  {t:15s}: {n:4d} leads")
print()
print("Paying customers by tier (sanity check — should be mostly Strong/Likely):")
for t in ["Strong ICP","Likely ICP","Weak ICP","Not ICP"]:
    n = paying_tier.get(t, 0)
    print(f"  {t:15s}: {n:3d} paying customers")
print()
print("Top ICP candidates (non-paying, by score):")
for _, r in candidates.head(20).iterrows():
    co  = str(r.get("company",""))[:35]
    ind = str(r.get("industry",""))[:20]
    oc  = str(r.get("outcome_class",""))
    print(f"  {co:35s} {ind:20s} [{oc:11s}]  score={r['icp_v3_score']:.0f}  tier={r['icp_v3_tier']}")

print("\nDone.")
