"""
Churn Analysis — paying vs churned ICP comparison
Questions:
  1. How similar are churned companies to paying customers?
  2. Which industries / sizes / functions churn most?
  3. Are there distinct churn sub-segments? (product-gap churners vs poor-fit churners)
  4. Which churned companies are worth re-engaging?
"""

import pandas as pd
import numpy as np
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

BG     = "#0f1117"
CARD   = "#1a1d2e"
TEXT   = "#e0e0e0"
GREEN  = "#22c55e"
PURPLE = "#6c63ff"
YELLOW = "#f59e0b"
RED    = "#ef4444"
GRAY   = "#4b5563"
BLUE   = "#3b82f6"

# ── Load ─────────────────────────────────────────────────────────────────────

df  = pd.read_csv("lead_scoring_features.csv")

# Deduplicate by company — keep the row with the highest sale_stage_score
# (most information), or first row if tied.
df = df.sort_values("sale_stage_score", ascending=False).drop_duplicates(subset=["company"], keep="first").reset_index(drop=True)

# Merge v3 scores — also deduplicate v3 to one row per company (best score)
v3 = pd.read_csv("icp_v3_scores.csv")[["company","icp_v3_score","icp_v3_tier"]]
v3 = v3.sort_values("icp_v3_score", ascending=False).drop_duplicates(subset=["company"], keep="first")
df = df.merge(v3, on="company", how="left")

for col in df.columns:
    if df[col].dtype == object and df[col].isin(["True","False"]).any():
        df[col] = df[col].map({"True":1,"False":0,True:1,False:0})

paying  = df[df["converted"] == 1].copy()
churned = df[df["outcome_class"] == "canceled"].copy()
both    = pd.concat([paying, churned])
both["group"] = both.apply(lambda r: "Paying" if r["converted"]==1 else "Churned", axis=1)

print(f"Paying:  {len(paying)}")
print(f"Churned: {len(churned)}")

# ── Helper: industry from one-hot ─────────────────────────────────────────────

IND_MAP = {
    "ind_consulting":    "Consulting",
    "ind_education":     "Education",
    "ind_finance":       "Finance",
    "ind_government":    "Government",
    "ind_healthcare":    "Healthcare",
    "ind_hospitality":   "Hospitality",
    "ind_manufacturing": "Manufacturing",
    "ind_media":         "Media",
    "ind_realestate":    "Real Estate",
    "ind_retail":        "Retail",
    "ind_tech":          "Tech / IT",
    "ind_other":         "Other",
}

def get_industry(row):
    for col, label in IND_MAP.items():
        if col in row and row[col] == 1:
            return label
    return "Other"

SEN_MAP = {
    "sen_c_suite":"C-Suite","sen_vp":"VP","sen_director":"Director",
    "sen_manager":"Manager","sen_ic":"Individual Contributor","sen_unknown":"Unknown",
}

def get_seniority(row):
    for col, label in SEN_MAP.items():
        if col in row and row[col] == 1:
            return label
    return "Unknown"

for grp in [paying, churned, both]:
    grp["industry_label"] = grp.apply(get_industry, axis=1)
    grp["seniority_label"] = grp.apply(get_seniority, axis=1)

# Size bucket
def size_bucket(sz):
    if pd.isna(sz): return "Unknown"
    if sz < 50:    return "Micro\n(<50)"
    if sz < 200:   return "Small\n(50–199)"
    if sz < 1500:  return "Mid\n(200–1499)"
    if sz < 5000:  return "Large\n(1500–4999)"
    return "Mega\n(5000+)"

paying["size_bucket"]  = paying["company_size"].apply(size_bucket)
churned["size_bucket"] = churned["company_size"].apply(size_bucket)

# ── Print raw breakdowns ──────────────────────────────────────────────────────

print("\n--- INDUSTRY ---")
print("Paying:")
print(paying["industry_label"].value_counts().to_string())
print("\nChurned:")
print(churned["industry_label"].value_counts().to_string())

print("\n--- SIZE BUCKET ---")
sz_order = ["Micro\n(<50)","Small\n(50–199)","Mid\n(200–1499)","Large\n(1500–4999)","Mega\n(5000+)","Unknown"]
print("Paying:")
print(paying["size_bucket"].value_counts().to_string())
print("\nChurned:")
print(churned["size_bucket"].value_counts().to_string())

print("\n--- SENIORITY ---")
print("Paying:")
print(paying["seniority_label"].value_counts().to_string())
print("\nChurned:")
print(churned["seniority_label"].value_counts().to_string())

# ── Churn sub-segment clustering ─────────────────────────────────────────────

FEATURES = [
    "sz_mid","sz_small","sz_large","sz_enterprise","sz_mega","sz_micro",
    "company_size","ind_finance","ind_government","ind_consulting","ind_education",
    "ind_manufacturing","ind_healthcare","ind_media","ind_realestate","ind_tech","ind_other",
    "sen_c_suite","sen_director","sen_vp","sen_manager","sen_ic","seniority_score",
]
feats = [f for f in FEATURES if f in churned.columns]
X_ch  = churned[feats].fillna(0).astype(float)

scaler = StandardScaler()
X_s    = scaler.fit_transform(X_ch)

km = KMeans(n_clusters=3, random_state=42, n_init=20)
churned["churn_segment"] = km.fit_predict(X_s)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_s)
churned["pca1"] = X_pca[:,0]
churned["pca2"] = X_pca[:,1]

# Also PCA for paying
X_pay = paying[feats].fillna(0).astype(float)
X_pay_s = scaler.transform(X_pay)
X_pay_pca = pca.transform(X_pay_s)
paying["pca1"] = X_pay_pca[:,0]
paying["pca2"] = X_pay_pca[:,1]

print("\n--- CHURN SEGMENTS ---")
for seg in sorted(churned["churn_segment"].unique()):
    sub = churned[churned["churn_segment"]==seg]
    avg_sz  = sub["company_size"].median()
    top_ind = sub["industry_label"].value_counts().head(3).to_dict()
    top_sen = sub["seniority_label"].value_counts().head(2).to_dict()
    avg_icp = sub["icp_v3_score"].mean() if "icp_v3_score" in sub.columns else 0
    print(f"\nSegment {seg} (n={len(sub)}):")
    print(f"  Median size: {avg_sz:.0f} employees")
    print(f"  Top industries: {top_ind}")
    print(f"  Top seniority: {top_sen}")
    print(f"  Avg ICP v3 score: {avg_icp:.1f}")
    print(f"  Companies: {', '.join(sub['company'].fillna('?').astype(str).head(8).tolist())}")

# ── Build figures ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 18), facecolor=BG)
gs  = GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.38)

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(CARD)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(colors=TEXT, labelsize=8)
    if title:  ax.set_title(title,  color=TEXT, fontsize=10, fontweight="bold", pad=8)
    if xlabel: ax.set_xlabel(xlabel, color=TEXT, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, color=TEXT, fontsize=8)
    ax.xaxis.grid(True, color="#2a2d3e", zorder=0)
    ax.set_axisbelow(True)

# 1. Industry: paying vs churned side-by-side
ax1 = fig.add_subplot(gs[0, :2])
style_ax(ax1, "Industry — Paying vs Churned (% of each group)")
all_inds = list(IND_MAP.values())
pay_pct  = paying["industry_label"].value_counts(normalize=True).reindex(all_inds, fill_value=0) * 100
ch_pct   = churned["industry_label"].value_counts(normalize=True).reindex(all_inds, fill_value=0) * 100
x = np.arange(len(all_inds))
w = 0.38
bars_p = ax1.barh(x + w/2, pay_pct.values, height=w, color=GREEN,  alpha=0.85, label="Paying",  zorder=3)
bars_c = ax1.barh(x - w/2, ch_pct.values,  height=w, color=YELLOW, alpha=0.85, label="Churned", zorder=3)
ax1.set_yticks(x)
ax1.set_yticklabels(all_inds, color=TEXT, fontsize=8)
ax1.set_xlabel("% of group", color=TEXT, fontsize=8)
ax1.legend(fontsize=8, facecolor=CARD, labelcolor=TEXT, framealpha=0.8)
# Annotate differences
for i, ind in enumerate(all_inds):
    diff = pay_pct[ind] - ch_pct[ind]
    if abs(diff) >= 3:
        color = GREEN if diff > 0 else RED
        ax1.text(max(pay_pct[ind], ch_pct[ind]) + 0.5, i,
                 f"{'↑' if diff>0 else '↓'}{abs(diff):.0f}pp", va="center",
                 color=color, fontsize=7.5, fontweight="bold")

# 2. Size distribution
ax2 = fig.add_subplot(gs[0, 2])
style_ax(ax2, "Company Size Distribution")
sz_order_short = ["Micro\n(<50)","Small\n(50–199)","Mid\n(200–1499)","Large\n(1500–4999)","Mega\n(5000+)"]
pay_sz = paying["size_bucket"].value_counts(normalize=True).reindex(sz_order_short, fill_value=0)*100
ch_sz  = churned["size_bucket"].value_counts(normalize=True).reindex(sz_order_short, fill_value=0)*100
x2 = np.arange(len(sz_order_short))
ax2.barh(x2 + 0.2, pay_sz.values, height=0.38, color=GREEN,  alpha=0.85, label="Paying",  zorder=3)
ax2.barh(x2 - 0.2, ch_sz.values,  height=0.38, color=YELLOW, alpha=0.85, label="Churned", zorder=3)
ax2.set_yticks(x2)
ax2.set_yticklabels([s.replace("\n"," ") for s in sz_order_short], color=TEXT, fontsize=8)
ax2.set_xlabel("% of group", color=TEXT, fontsize=8)
ax2.legend(fontsize=7.5, facecolor=CARD, labelcolor=TEXT, framealpha=0.8)

# 3. Seniority distribution
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, "Seniority — Paying vs Churned")
sen_order = ["C-Suite","VP","Director","Manager","Individual Contributor","Unknown"]
pay_sen = paying["seniority_label"].value_counts(normalize=True).reindex(sen_order, fill_value=0)*100
ch_sen  = churned["seniority_label"].value_counts(normalize=True).reindex(sen_order, fill_value=0)*100
x3 = np.arange(len(sen_order))
ax3.barh(x3 + 0.2, pay_sen.values, height=0.38, color=GREEN,  alpha=0.85, label="Paying",  zorder=3)
ax3.barh(x3 - 0.2, ch_sen.values,  height=0.38, color=YELLOW, alpha=0.85, label="Churned", zorder=3)
ax3.set_yticks(x3)
ax3.set_yticklabels(sen_order, color=TEXT, fontsize=8)
ax3.set_xlabel("% of group", color=TEXT, fontsize=8)
ax3.legend(fontsize=7.5, facecolor=CARD, labelcolor=TEXT, framealpha=0.8)

# 4. PCA scatter — paying vs churned in same space
ax4 = fig.add_subplot(gs[1, 1:])
style_ax(ax4, "PCA Landscape — Where Churned vs Paying Sit in Feature Space")
ax4.scatter(churned["pca1"], churned["pca2"],
            c=YELLOW, s=80, alpha=0.75, zorder=3, label="Churned", marker="D", edgecolors="#0f1117", linewidths=0.4)
ax4.scatter(paying["pca1"], paying["pca2"],
            c=GREEN, s=140, alpha=0.9, zorder=4, label="Paying", marker="*", edgecolors="white", linewidths=0.5)
# Add centroids
p_cx, p_cy = paying["pca1"].mean(), paying["pca2"].mean()
c_cx, c_cy = churned["pca1"].mean(), churned["pca2"].mean()
ax4.scatter([p_cx], [p_cy], c="white", s=250, marker="X", zorder=5, label="Paying centroid")
ax4.scatter([c_cx], [c_cy], c=RED,     s=250, marker="X", zorder=5, label="Churned centroid")
ax4.annotate("Paying\ncentroid", (p_cx, p_cy), textcoords="offset points", xytext=(8, 6),
             color="white", fontsize=7.5, fontweight="bold")
ax4.annotate("Churned\ncentroid", (c_cx, c_cy), textcoords="offset points", xytext=(8, -14),
             color=RED, fontsize=7.5, fontweight="bold")
ax4.xaxis.grid(True, color="#2a2d3e")
ax4.yaxis.grid(True, color="#2a2d3e")
ax4.set_xlabel("PCA 1", color=TEXT, fontsize=8)
ax4.set_ylabel("PCA 2", color=TEXT, fontsize=8)
ax4.legend(fontsize=7.5, facecolor=CARD, labelcolor=TEXT, framealpha=0.8)

# 5. Churn sub-segments
ax5 = fig.add_subplot(gs[2, :2])
style_ax(ax5, "Churn Sub-Segments (K-Means k=3) — Who Churned and Why?")
seg_colors = [BLUE, YELLOW, PURPLE]
seg_labels_map = {}
for seg in sorted(churned["churn_segment"].unique()):
    sub = churned[churned["churn_segment"]==seg]
    top_ind = sub["industry_label"].value_counts().index[0] if len(sub) > 0 else "?"
    med_sz  = sub["company_size"].median()
    top_sen = sub["seniority_label"].value_counts().index[0] if len(sub) > 0 else "?"
    lbl = f"Seg {seg} (n={len(sub)})\n{top_ind}, ~{med_sz:.0f} emp, {top_sen}"
    seg_labels_map[seg] = lbl
    ax5.scatter(sub["pca1"], sub["pca2"],
                c=seg_colors[seg % 3], s=90, alpha=0.8, zorder=3, label=lbl,
                edgecolors="#0f1117", linewidths=0.4)
ax5.scatter(paying["pca1"], paying["pca2"],
            c=GREEN, s=140, alpha=0.9, zorder=4, label="Paying customers ✓",
            marker="*", edgecolors="white", linewidths=0.5)
ax5.xaxis.grid(True, color="#2a2d3e")
ax5.yaxis.grid(True, color="#2a2d3e")
ax5.set_xlabel("PCA 1", color=TEXT, fontsize=8)
ax5.set_ylabel("PCA 2", color=TEXT, fontsize=8)
ax5.legend(fontsize=7.5, facecolor=CARD, labelcolor=TEXT, framealpha=0.8, loc="upper right")

# 6. Re-engagement priority table
ax6 = fig.add_subplot(gs[2, 2])
ax6.set_facecolor(CARD)
for sp in ax6.spines.values(): sp.set_visible(False)
ax6.axis("off")

if "icp_v3_score" in churned.columns:
    reeng = churned.dropna(subset=["icp_v3_score"]).sort_values("icp_v3_score", ascending=False).head(10)
    tdata = []
    for _, r in reeng.iterrows():
        co  = str(r.get("company",""))[:18]
        ind = str(r.get("industry_label",""))[:12]
        sz  = f"{r['company_size']:.0f}" if pd.notna(r.get("company_size")) else "?"
        sc  = f"{r['icp_v3_score']:.0f}"
        tdata.append([co, ind, sz, sc])
    tbl = ax6.table(cellText=tdata,
                    colLabels=["Company", "Industry", "Size", "Score"],
                    bbox=[0, 0, 1, 1], cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    for (ri, ci), cell in tbl.get_celld().items():
        cell.set_edgecolor("#2a2d3e")
        if ri == 0:
            cell.set_facecolor(PURPLE)
            cell.set_text_props(color="white", fontweight="bold")
        else:
            cell.set_facecolor("#242738" if ri % 2 == 0 else CARD)
            if ci == 3:
                sc_val = float(tdata[ri-1][3]) if ri <= len(tdata) else 0
                cell.set_text_props(color=GREEN if sc_val >= 75 else YELLOW, fontweight="bold")
            else:
                cell.set_text_props(color=TEXT)
    ax6.set_title("Re-engagement Priority\n(Churned, by ICP v3 Score)", color=TEXT, fontsize=9, fontweight="bold", pad=8)

fig.suptitle("LEAD.bot Churn Analysis — Paying vs Churned ICP Profile\n"
             "Both groups bought the vision. Churn = product maturity gap, not ICP mismatch.",
             color=TEXT, fontsize=13, fontweight="bold", y=0.99)

plt.savefig("churn_analysis.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("\nSaved: churn_analysis.png")

# ── Segment summary ───────────────────────────────────────────────────────────

print("\n" + "="*65)
print("CHURN SEGMENT PROFILES")
print("="*65)

for seg in sorted(churned["churn_segment"].unique()):
    sub = churned[churned["churn_segment"]==seg]
    print(f"\nSegment {seg}  (n={len(sub)}, avg ICP={sub['icp_v3_score'].mean():.1f})")
    print(f"  Median size:    {sub['company_size'].median():.0f} employees")
    print(f"  Size range:     {sub['company_size'].min():.0f} – {sub['company_size'].max():.0f}")
    print(f"  Top industries: {dict(sub['industry_label'].value_counts().head(4))}")
    print(f"  Top seniority:  {dict(sub['seniority_label'].value_counts().head(3))}")
    print(f"  Companies:")
    for _, r in sub.sort_values("icp_v3_score", ascending=False).iterrows():
        sz  = f"{r['company_size']:.0f}" if pd.notna(r["company_size"]) else "?"
        ind = r["industry_label"]
        sc  = f"{r['icp_v3_score']:.0f}" if pd.notna(r.get("icp_v3_score")) else "?"
        print(f"    {str(r['company']):35s} {ind:20s} {sz:>8} emp  ICP={sc}")

print("\n\nRE-ENGAGEMENT PRIORITY (churned, sorted by ICP v3):")
if "icp_v3_score" in churned.columns:
    for _, r in churned.sort_values("icp_v3_score", ascending=False).iterrows():
        sz = f"{r['company_size']:.0f}" if pd.notna(r["company_size"]) else "?"
        print(f"  {str(r['company']):35s} {r['industry_label']:20s} {sz:>8} emp  ICP={r['icp_v3_score']:.0f}  seg={r['churn_segment']}")
