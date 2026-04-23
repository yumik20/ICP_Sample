"""
Cohort Analysis for LEAD.bot Lead Scoring
Generates cohort_analysis.png and cohort_heatmap.png
Segments: Industry, Company Size, Seniority, Region, Funnel Stage
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

df = pd.read_csv("lead_scoring_features.csv")

BASELINE = df["converted"].mean()  # ~3.1%
ACCENT   = "#6c63ff"
BG       = "#0f1117"
CARD_BG  = "#1a1d2e"
TEXT     = "#e0e0e0"
GREEN    = "#22c55e"
RED      = "#ef4444"
YELLOW   = "#f59e0b"
GRAY     = "#4b5563"

def lift_color(rate, baseline=BASELINE):
    lift = rate / baseline if baseline > 0 else 1
    if lift >= 2.5:  return GREEN
    if lift >= 1.5:  return ACCENT
    if lift >= 0.8:  return YELLOW
    return RED

# ── Build cohort tables ───────────────────────────────────────────────────────

def cohort_table(prefix, label_map):
    rows = []
    cols = [c for c in df.columns if c.startswith(prefix)]
    for col in cols:
        key = col[len(prefix):]
        label = label_map.get(key, key.replace("_", " ").title())
        n = int(df[col].sum())
        conv = int(df[df[col] == 1]["converted"].sum())
        rate = conv / n if n > 0 else 0
        rows.append({"label": label, "n": n, "converted": conv, "rate": rate})
    return pd.DataFrame(rows).sort_values("rate", ascending=False)

IND_LABELS = {
    "consulting": "Consulting", "education": "Education", "finance": "Finance",
    "government": "Government", "healthcare": "Healthcare", "hospitality": "Hospitality",
    "manufacturing": "Manufacturing", "media": "Media / Marketing",
    "other": "Other / Unknown", "realestate": "Real Estate", "retail": "Retail", "tech": "Tech / SaaS",
}
SZ_LABELS = {
    "micro": "Micro\n(<50)", "small": "Small\n(50–199)", "mid": "Mid\n(200–1499)",
    "large": "Large\n(1500–4999)", "enterprise": "Enterprise\n(5000–9999)",
    "mega": "Mega\n(10000+)", "unknown": "Unknown",
}
SEN_LABELS = {
    "c_suite": "C-Suite", "vp": "VP", "director": "Director",
    "manager": "Manager", "ic": "Individual\nContrib.", "unknown": "Unknown",
}
REG_LABELS = {
    "north_america": "N. America", "europe": "Europe", "apac": "APAC",
    "latam": "LATAM", "other": "Other", "unknown": "Unknown",
}

ind_df  = cohort_table("ind_",  IND_LABELS)
sz_df   = cohort_table("sz_",   SZ_LABELS)
sen_df  = cohort_table("sen_",  SEN_LABELS)
reg_df  = cohort_table("reg_",  REG_LABELS)

# Funnel stage breakdown (outcome_class × industry top 6)
stage_counts = df["outcome_class"].value_counts().reindex(["paying","in_progress","inactive","canceled","unknown"]).fillna(0)

# ── Figure 1: Main cohort chart ───────────────────────────────────────────────

fig = plt.figure(figsize=(18, 14), facecolor=BG)
gs  = GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.4)

def bar_cohort(ax, cdf, title, baseline=BASELINE, show_n=True):
    ax.set_facecolor(CARD_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=TEXT, labelsize=8)

    colors = [lift_color(r) for r in cdf["rate"]]
    bars = ax.barh(cdf["label"], cdf["rate"] * 100, color=colors, height=0.6, zorder=3)
    ax.axvline(baseline * 100, color=YELLOW, linewidth=1.2, linestyle="--", zorder=4, label=f"Baseline {baseline*100:.1f}%")

    for bar, (_, row) in zip(bars, cdf.iterrows()):
        pct = row["rate"] * 100
        label_txt = f'{pct:.1f}%  (n={row["n"]})' if show_n else f'{pct:.1f}%'
        ax.text(pct + 0.3, bar.get_y() + bar.get_height() / 2,
                label_txt, va="center", ha="left", color=TEXT, fontsize=7.5)

    ax.set_xlim(0, max(cdf["rate"].max() * 100 * 1.45, baseline * 100 * 2))
    ax.set_xlabel("Conversion Rate (%)", color=TEXT, fontsize=8)
    ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=8)
    ax.xaxis.grid(True, color="#2a2d3e", zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, facecolor=CARD_BG, labelcolor=TEXT, framealpha=0.8)

# 1. Industry (filter out zero-n and "other" for clarity)
ax1 = fig.add_subplot(gs[0, :2])
ind_plot = ind_df[ind_df["n"] >= 5].copy()
bar_cohort(ax1, ind_plot, "Industry Cohort — Conversion Rate vs Baseline")

# 2. Company Size
ax2 = fig.add_subplot(gs[0, 2])
sz_plot = sz_df[sz_df["label"] != "Unknown"].copy()
bar_cohort(ax2, sz_plot, "Company Size Cohort", show_n=True)

# 3. Seniority
ax3 = fig.add_subplot(gs[1, :2])
sen_plot = sen_df.copy()
bar_cohort(ax3, sen_plot, "Seniority Cohort — Conversion Rate vs Baseline")

# 4. Region
ax4 = fig.add_subplot(gs[1, 2])
reg_plot = reg_df[reg_df["n"] >= 3].copy()
bar_cohort(ax4, reg_plot, "Region Cohort", show_n=True)

# 5. Funnel funnel stage donut
ax5 = fig.add_subplot(gs[2, 0])
ax5.set_facecolor(CARD_BG)
for sp in ax5.spines.values(): sp.set_visible(False)
labels = ["Paying", "In Progress", "Inactive", "Canceled"]
sizes  = [stage_counts.get(k, 0) for k in ["paying","in_progress","inactive","canceled"]]
colors_pie = [GREEN, ACCENT, GRAY, RED]
wedges, texts, autotexts = ax5.pie(
    sizes, labels=None, colors=colors_pie, autopct="%1.0f%%",
    startangle=90, pctdistance=0.75,
    wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2),
)
for at in autotexts: at.set(color=BG, fontsize=8, fontweight="bold")
ax5.legend(wedges, [f"{l} ({int(s)})" for l, s in zip(labels, sizes)],
           loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=2,
           fontsize=7.5, facecolor=CARD_BG, labelcolor=TEXT, framealpha=0.8)
ax5.set_title("Funnel Stage Breakdown", color=TEXT, fontsize=10, fontweight="bold", pad=8)

# 6. Lift table — top segments
ax6 = fig.add_subplot(gs[2, 1:])
ax6.set_facecolor(CARD_BG)
for sp in ax6.spines.values(): sp.set_visible(False)
ax6.axis("off")

segments = []
for cdf, cat in [(ind_df, "Industry"), (sz_df, "Size"), (sen_df, "Seniority"), (reg_df, "Region")]:
    for _, row in cdf.iterrows():
        if row["n"] >= 5 and row["label"] not in ("Unknown", "Other / Unknown", "Other"):
            segments.append({"Category": cat, "Segment": row["label"].replace("\n"," "),
                             "N": row["n"], "Converted": row["converted"],
                             "Rate": row["rate"], "Lift": row["rate"] / BASELINE})

seg_df = pd.DataFrame(segments).sort_values("Lift", ascending=False).head(10)

col_labels = ["Category", "Segment", "N", "Converted", "Conv %", "Lift vs\nBaseline"]
table_data = []
for _, r in seg_df.iterrows():
    table_data.append([r["Category"], r["Segment"], int(r["N"]), int(r["Converted"]),
                       f"{r['Rate']*100:.1f}%", f"{r['Lift']:.1f}×"])

tbl = ax6.table(cellText=table_data, colLabels=col_labels,
                bbox=[0, 0, 1, 1], cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)

for (row_i, col_i), cell in tbl.get_celld().items():
    cell.set_edgecolor("#2a2d3e")
    if row_i == 0:
        cell.set_facecolor(ACCENT)
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("#242738" if row_i % 2 == 0 else CARD_BG)
        cell.set_text_props(color=TEXT)
        # Color the lift column
        if col_i == 5:
            try:
                lift_val = float(table_data[row_i - 1][5].replace("×",""))
                cell.set_text_props(color=lift_color(lift_val * BASELINE), fontweight="bold")
            except:
                pass

ax6.set_title("Top 10 Segments by Lift", color=TEXT, fontsize=10, fontweight="bold", pad=8)

# Legend for bar colors
legend_patches = [
    mpatches.Patch(color=GREEN,  label="≥2.5× lift (strong)"),
    mpatches.Patch(color=ACCENT, label="1.5–2.5× lift (good)"),
    mpatches.Patch(color=YELLOW, label="0.8–1.5× lift (neutral)"),
    mpatches.Patch(color=RED,    label="<0.8× lift (weak)"),
]
fig.legend(handles=legend_patches, loc="lower center", ncol=4, fontsize=8,
           facecolor=CARD_BG, labelcolor=TEXT, framealpha=0.9,
           bbox_to_anchor=(0.5, 0.01))

fig.suptitle("LEAD.bot Cohort Analysis — Conversion Lift by Segment",
             color=TEXT, fontsize=14, fontweight="bold", y=0.98)

plt.savefig("cohort_analysis.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("Saved cohort_analysis.png")

# ── Figure 2: Cross-segment heatmap ──────────────────────────────────────────

fig2, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
fig2.suptitle("LEAD.bot Cross-Segment Heatmap — Conversion Rate %",
              color=TEXT, fontsize=13, fontweight="bold")

def cross_heatmap(ax, row_prefix, col_prefix, row_labels, col_labels, title):
    ax.set_facecolor(CARD_BG)
    r_cols = [c for c in df.columns if c.startswith(row_prefix)]
    c_cols = [c for c in df.columns if c.startswith(col_prefix)]

    matrix = np.zeros((len(r_cols), len(c_cols)))
    n_matrix = np.zeros((len(r_cols), len(c_cols)), dtype=int)

    for i, rc in enumerate(r_cols):
        for j, cc in enumerate(c_cols):
            mask = (df[rc] == 1) & (df[cc] == 1)
            n = mask.sum()
            conv = df.loc[mask, "converted"].sum()
            matrix[i, j] = conv / n * 100 if n >= 3 else np.nan
            n_matrix[i, j] = n

    r_labels = [row_labels.get(c[len(row_prefix):], c[len(row_prefix):]) for c in r_cols]
    c_labels = [col_labels.get(c[len(col_prefix):], c[len(col_prefix):]) for c in c_cols]

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=20)
    ax.set_xticks(range(len(c_cols)))
    ax.set_yticks(range(len(r_cols)))
    ax.set_xticklabels(c_labels, color=TEXT, fontsize=8, rotation=30, ha="right")
    ax.set_yticklabels(r_labels, color=TEXT, fontsize=8)
    ax.tick_params(colors=TEXT)

    for i in range(len(r_cols)):
        for j in range(len(c_cols)):
            val = matrix[i, j]
            n   = n_matrix[i, j]
            if not np.isnan(val):
                txt = f"{val:.0f}%\nn={n}"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if val < 10 else "black",
                        fontsize=6.5, fontweight="bold")
            else:
                ax.text(j, i, "—", ha="center", va="center", color=GRAY, fontsize=9)

    plt.colorbar(im, ax=ax, label="Conversion %", fraction=0.04)
    ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=10)
    for sp in ax.spines.values(): sp.set_visible(False)

SEN_SHORT = {"c_suite":"C-Suite","vp":"VP","director":"Director","manager":"Manager","ic":"IC","unknown":"Unknown"}
IND_SHORT  = {"consulting":"Consult.","education":"Edu.","finance":"Finance","government":"Gov.",
               "healthcare":"Health.","manufacturing":"Mfg.","media":"Media","other":"Other",
               "realestate":"RE","retail":"Retail","tech":"Tech","hospitality":"Hosp."}
SZ_SHORT   = {"micro":"<50","small":"50-199","mid":"200-1499","large":"1500-4999","enterprise":"5K-10K","mega":"10K+","unknown":"Unk."}

cross_heatmap(axes[0], "sen_", "ind_", SEN_SHORT, IND_SHORT, "Seniority × Industry Conversion Rate")
cross_heatmap(axes[1], "sz_",  "ind_", SZ_SHORT,  IND_SHORT, "Company Size × Industry Conversion Rate")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("cohort_heatmap.png", dpi=150, bbox_inches="tight", facecolor=BG)
print("Saved cohort_heatmap.png")

# ── Print summary ─────────────────────────────────────────────────────────────
print(f"\nBaseline conversion rate: {BASELINE*100:.2f}%")
print("\nTop cohorts by lift:")
for _, r in seg_df.iterrows():
    print(f"  {r['Category']:12s} {r['Segment']:20s}  {r['Rate']*100:.1f}%  ({r['Lift']:.1f}× lift)")
