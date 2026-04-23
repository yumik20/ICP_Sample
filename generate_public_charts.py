"""
Regenerate charts for public showcase:
- Fake industry sector names
- No paying/churned/canceled labels — only "ICP Confirmed" vs "Unlabeled"
- No exact counts visible as business data
- Neutral, methodology-focused language
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
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

# ── Fake industry name mapping ────────────────────────────────────────────────

FAKE_IND = {
    "ind_finance":       "Regulated Services",
    "ind_government":    "Public Institutions",
    "ind_consulting":    "Professional Services",
    "ind_education":     "Academic & Research",
    "ind_healthcare":    "Life Sciences",
    "ind_manufacturing": "Industrial & Engineering",
    "ind_media":         "Media & Communications",
    "ind_realestate":    "Property & Infrastructure",
    "ind_retail":        "Consumer & Retail",
    "ind_tech":          "Technology",
    "ind_other":         "Other Sectors",
}

FAKE_SZ = {
    "sz_micro":      "XS  (<50)",
    "sz_small":      "S   (50–199)",
    "sz_mid":        "M   (200–1499)",
    "sz_large":      "L   (1500–4999)",
    "sz_enterprise": "XL  (5000–9999)",
    "sz_mega":       "XXL (10000+)",
}

FAKE_SEN = {
    "sen_c_suite":  "Executive",
    "sen_vp":       "Vice President",
    "sen_director": "Director",
    "sen_manager":  "Manager",
    "sen_ic":       "Individual Contributor",
    "sen_unknown":  "Unknown",
}

FAKE_REG = {
    "reg_north_america": "Region A",
    "reg_europe":        "Region B",
    "reg_apac":          "Region C",
    "reg_latam":         "Region D",
    "reg_other":         "Region E",
    "reg_unknown":       "Unknown",
}

# ── Load & prep ───────────────────────────────────────────────────────────────

df = pd.read_csv("lead_scoring_features.csv")
for col in df.columns:
    if df[col].dtype == object and df[col].isin(["True","False"]).any():
        df[col] = df[col].map({"True":1,"False":0,True:1,False:0})

# ICP confirmed = paying + churned; everything else = unlabeled
icp_confirmed = (df["converted"] == 1) | (df["outcome_class"] == "canceled")
unlabeled     = ~icp_confirmed
df["icp_label"] = icp_confirmed.map({True:"ICP Confirmed", False:"Unlabeled"})

BASELINE = df.loc[icp_confirmed].shape[0] / len(df)  # ratio for lift calc

# ── Chart 1: Public cohort analysis ──────────────────────────────────────────

fig = plt.figure(figsize=(18, 14), facecolor=BG)
gs  = GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.40)

def style_ax(ax, title=""):
    ax.set_facecolor(CARD)
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.tick_params(colors=TEXT, labelsize=8)
    if title: ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=8)
    ax.xaxis.grid(True, color="#2a2d3e", zorder=0)
    ax.set_axisbelow(True)

def lift_color(rate, baseline=BASELINE):
    lift = rate / baseline if baseline > 0 else 1
    if lift >= 2.5:  return GREEN
    if lift >= 1.5:  return PURPLE
    if lift >= 0.8:  return YELLOW
    return RED

def cohort_bars(ax, prefix, label_map, title, show_n=False):
    style_ax(ax, title)
    rows = []
    for col, label in label_map.items():
        if col not in df.columns: continue
        mask = df[col] == 1
        n    = mask.sum()
        conv = df.loc[mask & icp_confirmed].shape[0]
        rate = conv / n if n > 0 else 0
        rows.append({"label": label, "n": n, "conv": conv, "rate": rate})
    cdf = pd.DataFrame(rows).sort_values("rate", ascending=False)
    colors = [lift_color(r) for r in cdf["rate"]]
    ax.barh(cdf["label"], cdf["rate"]*100, color=colors, height=0.6, zorder=3)
    ax.axvline(BASELINE*100, color=YELLOW, linewidth=1.2, linestyle="--", zorder=4,
               label=f"Baseline {BASELINE*100:.1f}%")
    for i, (_, row) in enumerate(cdf.iterrows()):
        pct = row["rate"]*100
        ax.text(pct+0.3, i, f"{pct:.1f}%", va="center", ha="left", color=TEXT, fontsize=7.5)
    ax.set_xlim(0, max(cdf["rate"].max()*100*1.5, BASELINE*100*2.5))
    ax.set_xlabel("ICP-confirmed rate (%)", color=TEXT, fontsize=8)
    ax.legend(fontsize=7, facecolor=CARD, labelcolor=TEXT, framealpha=0.8)

# Industry
ax1 = fig.add_subplot(gs[0, :2])
cohort_bars(ax1, "ind_", FAKE_IND, "Industry Sector — ICP Confirmation Rate vs Baseline")

# Size
ax2 = fig.add_subplot(gs[0, 2])
cohort_bars(ax2, "sz_", FAKE_SZ, "Company Size Band")

# Seniority
ax3 = fig.add_subplot(gs[1, :2])
cohort_bars(ax3, "sen_", FAKE_SEN, "Seniority — ICP Confirmation Rate vs Baseline")

# Region
ax4 = fig.add_subplot(gs[1, 2])
cohort_bars(ax4, "reg_", FAKE_REG, "Region")

# Funnel breakdown — rename groups
ax5 = fig.add_subplot(gs[2, 0])
ax5.set_facecolor(CARD)
for sp in ax5.spines.values(): sp.set_visible(False)
confirmed_n = icp_confirmed.sum()
unlabeled_n = unlabeled.sum()
wedges, _, autotexts = ax5.pie(
    [confirmed_n, unlabeled_n], labels=None,
    colors=[GREEN, GRAY], autopct="%1.0f%%", startangle=90, pctdistance=0.72,
    wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2)
)
for at in autotexts: at.set(color=BG, fontsize=9, fontweight="bold")
ax5.legend(wedges, ["ICP Confirmed", "Unlabeled / In-Progress"],
           loc="lower center", bbox_to_anchor=(0.5,-0.15), ncol=1,
           fontsize=8, facecolor=CARD, labelcolor=TEXT, framealpha=0.8)
ax5.set_title("Dataset Composition", color=TEXT, fontsize=10, fontweight="bold", pad=8)

# Top lift segments table
ax6 = fig.add_subplot(gs[2, 1:])
ax6.set_facecolor(CARD)
for sp in ax6.spines.values(): sp.set_visible(False)
ax6.axis("off")

all_segs = []
for prefix, label_map, cat in [
    ("ind_", FAKE_IND, "Industry"),
    ("sz_",  FAKE_SZ,  "Size"),
    ("sen_", FAKE_SEN, "Seniority"),
    ("reg_", FAKE_REG, "Region"),
]:
    for col, label in label_map.items():
        if col not in df.columns: continue
        mask = df[col] == 1
        n = mask.sum()
        conv = df.loc[mask & icp_confirmed].shape[0]
        rate = conv/n if n >= 5 else 0
        lift = rate/BASELINE if BASELINE > 0 else 1
        if n >= 5 and "Unknown" not in label:
            all_segs.append({"Category":cat,"Segment":label,"Rate":rate,"Lift":lift})

top = pd.DataFrame(all_segs).sort_values("Lift",ascending=False).head(10)
tdata = [[r["Category"],r["Segment"],f"{r['Rate']*100:.1f}%",f"{r['Lift']:.1f}×"]
         for _,r in top.iterrows()]
tbl = ax6.table(cellText=tdata, colLabels=["Category","Segment","ICP Rate","Lift"],
                bbox=[0,0,1,1], cellLoc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8)
for (ri,ci),cell in tbl.get_celld().items():
    cell.set_edgecolor("#2a2d3e")
    if ri==0:
        cell.set_facecolor(PURPLE); cell.set_text_props(color="white",fontweight="bold")
    else:
        cell.set_facecolor("#242738" if ri%2==0 else CARD)
        if ci==3:
            try:
                v = float(tdata[ri-1][3].replace("×",""))
                cell.set_text_props(color=GREEN if v>=2.5 else (PURPLE if v>=1.5 else YELLOW),
                                    fontweight="bold")
            except: pass
        else:
            cell.set_text_props(color=TEXT)
ax6.set_title("Top 10 Segments by Lift", color=TEXT, fontsize=10, fontweight="bold", pad=8)

legend_patches = [
    mpatches.Patch(color=GREEN,  label="≥2.5× lift"),
    mpatches.Patch(color=PURPLE, label="1.5–2.5× lift"),
    mpatches.Patch(color=YELLOW, label="0.8–1.5× lift"),
    mpatches.Patch(color=RED,    label="<0.8× lift"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=8,
           facecolor=CARD, labelcolor=TEXT, framealpha=0.9, bbox_to_anchor=(0.5,0.01))
fig.suptitle("Segment Lift Analysis — ICP Confirmation Rate by Firmographic Cohort",
             color=TEXT, fontsize=14, fontweight="bold", y=0.98)
plt.savefig("pub_cohort.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved pub_cohort.png")

# ── Chart 2: Public ICP v3 scoring chart ─────────────────────────────────────

ICP_FEATURES = [
    "sz_mid","sz_small","sz_large","sz_enterprise","sz_mega","sz_micro","company_size",
    "ind_finance","ind_government","ind_consulting","ind_education","ind_manufacturing",
    "ind_healthcare","ind_media","ind_realestate","ind_tech","ind_other",
    "sen_c_suite","sen_director","sen_vp","sen_manager","sen_ic","seniority_score",
    "days_since_last_contact",
]
features = [f for f in ICP_FEATURES if f in df.columns]
X = df[features].fillna(0).astype(float)
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

centroid = X_s[icp_confirmed].mean(axis=0)
euc_dist = np.linalg.norm(X_s - centroid, axis=1)
euc_sim  = 1/(1+euc_dist)
norms = np.linalg.norm(X_s, axis=1, keepdims=True); norms[norms==0]=1e-10
c_norm = centroid/(np.linalg.norm(centroid)+1e-10)
cos_sim = (X_s/norms) @ c_norm
raw = 0.6*cos_sim + 0.4*euc_sim
score = (raw-raw.min())/(raw.max()-raw.min())*100
df["pub_score"] = score

# PU model
threshold = np.percentile(score[unlabeled], 20)
proxy_neg  = unlabeled & (score <= threshold)
train_mask = icp_confirmed | proxy_neg
X_tr = X_s[train_mask]; y_tr = icp_confirmed[train_mask].astype(int).values
rf = RandomForestClassifier(n_estimators=300,class_weight="balanced",max_depth=6,random_state=42)
rf.fit(X_tr, y_tr)
pu = rf.predict_proba(X_s)[:,1]
pu_s = (pu-pu.min())/(pu.max()-pu.min())*100
iso = IsolationForest(n_estimators=200,contamination=0.3,random_state=42)
iso.fit(X_s[icp_confirmed])
iso_s_raw = iso.score_samples(X_s)
iso_s = (iso_s_raw-iso_s_raw.min())/(iso_s_raw.max()-iso_s_raw.min())*100
df["final_score"] = (0.4*score + 0.4*pu_s + 0.2*iso_s).round(1)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_s)
df["pca1"] = X_pca[:,0]; df["pca2"] = X_pca[:,1]

importances = pd.Series(rf.feature_importances_, index=features).sort_values(ascending=False)

# Rename features to fake names for display
FEAT_RENAME = {
    "sz_mid": "Size Band M (200–1499)",
    "sz_small": "Size Band S (50–199)",
    "sz_large": "Size Band L (1500–4999)",
    "sz_enterprise": "Size Band XL (5K–10K)",
    "sz_mega": "Size Band XXL (10K+)",
    "sz_micro": "Size Band XS (<50)",
    "company_size": "Headcount (raw)",
    "ind_finance": "Regulated Services",
    "ind_government": "Public Institutions",
    "ind_consulting": "Professional Services",
    "ind_education": "Academic & Research",
    "ind_manufacturing": "Industrial & Engineering",
    "ind_healthcare": "Life Sciences",
    "ind_media": "Media & Communications",
    "ind_realestate": "Property & Infrastructure",
    "ind_tech": "Technology",
    "ind_other": "Other Sectors",
    "sen_c_suite": "Executive",
    "sen_director": "Director",
    "sen_vp": "Vice President",
    "sen_manager": "Manager",
    "sen_ic": "Individual Contributor",
    "seniority_score": "Seniority Score",
    "days_since_last_contact": "Recency",
}
importances.index = [FEAT_RENAME.get(i, i) for i in importances.index]

fig2 = plt.figure(figsize=(18,16), facecolor=BG)
gs2  = GridSpec(3,3, figure=fig2, hspace=0.50, wspace=0.38)

# 1. Score distribution — ICP Confirmed vs Unlabeled
ax1 = fig2.add_subplot(gs2[0,:2])
ax1.set_facecolor(CARD)
for sp in ax1.spines.values(): sp.set_visible(False)
ax1.tick_params(colors=TEXT, labelsize=8)
bins = np.linspace(0,100,25)
ax1.hist(df.loc[unlabeled,"final_score"], bins=bins, color=GRAY, alpha=0.7, label="Unlabeled leads", zorder=3)
ax1.hist(df.loc[icp_confirmed,"final_score"], bins=bins, color=GREEN, alpha=0.9, label="ICP Confirmed ✓", zorder=4)
ax1.axvline(75, color=GREEN, linewidth=1, linestyle="--", alpha=0.7, label="Strong ICP threshold")
ax1.axvline(55, color=PURPLE, linewidth=1, linestyle="--", alpha=0.7, label="Likely ICP threshold")
ax1.set_xlabel("ICP Score", color=TEXT, fontsize=9)
ax1.set_ylabel("Count", color=TEXT, fontsize=9)
ax1.set_title("ICP Score Distribution — Confirmed vs Unlabeled Leads", color=TEXT, fontsize=10, fontweight="bold", pad=8)
ax1.legend(fontsize=8, facecolor=CARD, labelcolor=TEXT, framealpha=0.8)
ax1.xaxis.grid(True, color="#2a2d3e", zorder=0)

# 2. Tier pie
ax2 = fig2.add_subplot(gs2[0,2])
ax2.set_facecolor(CARD)
for sp in ax2.spines.values(): sp.set_visible(False)
def tier(s):
    if s>=75: return "Strong ICP"
    if s>=55: return "Likely ICP"
    if s>=35: return "Weak ICP"
    return "Low Signal"
df["pub_tier"] = df["final_score"].apply(tier)
tc = df["pub_tier"].value_counts().reindex(["Strong ICP","Likely ICP","Weak ICP","Low Signal"]).fillna(0)
w, _, at = ax2.pie(tc, labels=None, colors=[GREEN,PURPLE,YELLOW,RED],
    autopct="%1.0f%%", startangle=90, pctdistance=0.72,
    wedgeprops=dict(width=0.55,edgecolor=BG,linewidth=2))
for a in at: a.set(color=BG, fontsize=8, fontweight="bold")
ax2.legend(w, [f"{t}" for t in tc.index], loc="lower center", bbox_to_anchor=(0.5,-0.12),
           ncol=2, fontsize=7.5, facecolor=CARD, labelcolor=TEXT, framealpha=0.8)
ax2.set_title("Lead Tier Breakdown", color=TEXT, fontsize=10, fontweight="bold", pad=8)

# 3. PCA landscape
ax3 = fig2.add_subplot(gs2[1,:2])
ax3.set_facecolor(CARD)
for sp in ax3.spines.values(): sp.set_visible(False)
ax3.tick_params(colors=TEXT, labelsize=8)
sc = ax3.scatter(df.loc[unlabeled,"pca1"], df.loc[unlabeled,"pca2"],
                 c=df.loc[unlabeled,"final_score"], cmap="RdYlGn",
                 s=25, alpha=0.6, vmin=0, vmax=100, zorder=3)
ax3.scatter(df.loc[icp_confirmed,"pca1"], df.loc[icp_confirmed,"pca2"],
            c=GREEN, s=120, marker="*", zorder=5, label="ICP Confirmed ✓",
            edgecolors="white", linewidths=0.5)
plt.colorbar(sc, ax=ax3, label="ICP Score", fraction=0.04)
ax3.set_xlabel("Component 1", color=TEXT, fontsize=8)
ax3.set_ylabel("Component 2", color=TEXT, fontsize=8)
ax3.set_title("Lead Landscape — PCA Projection (colored by ICP Score)", color=TEXT, fontsize=10, fontweight="bold", pad=8)
ax3.legend(fontsize=8, facecolor=CARD, labelcolor=TEXT, framealpha=0.8)
ax3.xaxis.grid(True, color="#2a2d3e"); ax3.yaxis.grid(True, color="#2a2d3e")

# 4. Feature importance (renamed)
ax4 = fig2.add_subplot(gs2[1,2])
ax4.set_facecolor(CARD)
for sp in ax4.spines.values(): sp.set_visible(False)
ax4.tick_params(colors=TEXT, labelsize=8)
top_f = importances.head(12)
colors_f = [GREEN if v > importances.median() else PURPLE for v in top_f.values]
ax4.barh(top_f.index[::-1], top_f.values[::-1], color=colors_f[::-1], height=0.6, zorder=3)
ax4.set_xlabel("Importance", color=TEXT, fontsize=8)
ax4.set_title("Feature Importance (PU Random Forest)", color=TEXT, fontsize=10, fontweight="bold", pad=8)
ax4.xaxis.grid(True, color="#2a2d3e", zorder=0)
ax4.set_axisbelow(True)
ax4.tick_params(axis='y', labelsize=7.5)

# 5. Score by outcome group (renamed)
ax5 = fig2.add_subplot(gs2[2,:2])
ax5.set_facecolor(CARD)
for sp in ax5.spines.values(): sp.set_visible(False)
ax5.tick_params(colors=TEXT, labelsize=8)
groups = [
    ("ICP Confirmed", icp_confirmed, GREEN),
    ("Unlabeled / In-Progress", unlabeled, GRAY),
]
for i,(lbl,mask,col) in enumerate(groups):
    subset = df.loc[mask,"final_score"]
    ax5.boxplot(subset, positions=[i], widths=0.5, patch_artist=True,
                medianprops=dict(color="white",linewidth=2),
                boxprops=dict(facecolor=col,alpha=0.7),
                whiskerprops=dict(color=TEXT), capprops=dict(color=TEXT),
                flierprops=dict(marker="o",color=col,alpha=0.4,markersize=3))
ax5.set_xticks([0,1])
ax5.set_xticklabels(["ICP Confirmed","Unlabeled / In-Progress"], color=TEXT, fontsize=9)
ax5.set_ylabel("ICP Score", color=TEXT, fontsize=8)
ax5.set_ylim(0,105)
ax5.axhline(75, color=GREEN, linewidth=1, linestyle="--", alpha=0.5)
ax5.axhline(55, color=PURPLE, linewidth=1, linestyle="--", alpha=0.5)
ax5.set_title("Score Distribution by Group — Model Validation", color=TEXT, fontsize=10, fontweight="bold", pad=8)
ax5.xaxis.grid(False)
ax5.yaxis.grid(True, color="#2a2d3e", zorder=0)

# 6. Score examples
ax6 = fig2.add_subplot(gs2[2,2])
ax6.set_facecolor(CARD)
for sp in ax6.spines.values(): sp.set_visible(False)
ax6.axis("off")
examples = [
    ("Regulated Services, M-size, Director", 84, GREEN),
    ("Professional Services, S-size, Director", 78, GREEN),
    ("Academic & Research, M-size, Manager", 71, PURPLE),
    ("Life Sciences, L-size, VP", 62, PURPLE),
    ("Technology, XXL-size, IC", 31, RED),
]
ax6.set_title("Example Scores — Illustrative Profiles", color=TEXT, fontsize=9, fontweight="bold", pad=8)
for i,(label,score_val,col) in enumerate(examples):
    y = 0.82 - i*0.18
    ax6.text(0.02, y+0.04, label, transform=ax6.transAxes, color=TEXT, fontsize=7.5)
    bar_w = score_val/100 * 0.65
    ax6.add_patch(plt.Rectangle((0.02, y-0.05), 0.65, 0.07,
                                 facecolor="#2a2d4e", transform=ax6.transAxes, clip_on=False))
    ax6.add_patch(plt.Rectangle((0.02, y-0.05), bar_w, 0.07,
                                 facecolor=col, alpha=0.8, transform=ax6.transAxes, clip_on=False))
    ax6.text(0.70, y-0.01, f"{score_val}", transform=ax6.transAxes,
             color=col, fontsize=11, fontweight="bold", va="center")

fig2.suptitle("ICP Scoring Model — Positive-Unlabeled Learning\nLeads scored by similarity to confirmed ICP profile",
              color=TEXT, fontsize=13, fontweight="bold", y=0.99)
plt.savefig("pub_icp.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.close()
print("Saved pub_icp.png")

# ── Chart 3: Public model evaluation (relabel only) ───────────────────────────
# chart_model_evaluation.png is already clean — keep as-is
# shap_summary.png — feature names are fine (generic)
# icp_feature_importance.png — generic
print("Done. Public charts: pub_cohort.png, pub_icp.png")
print(f"\nICP Confirmed in dataset: {icp_confirmed.sum()} ({icp_confirmed.sum()/len(df)*100:.1f}%)")
print(f"Strong ICP leads: {(df['pub_tier']=='Strong ICP').sum()}")
print(f"Likely ICP leads: {(df['pub_tier']=='Likely ICP').sum()}")
