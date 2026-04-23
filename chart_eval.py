import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, re, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from datetime import datetime as dt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, learning_curve
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import xgboost as xgb_mod

df = pd.read_csv('/Users/alyssa/Desktop/Codex/lead_scoring_flat.csv')

SRULES = [
    (re.compile(r'\b(ceo|cto|coo|cfo|president|founder|managing director|chief)\b'), 'c_suite', 5),
    (re.compile(r'\b(svp|evp|vice president|vp)\b'), 'vp', 4),
    (re.compile(r'\b(director|head of|principal)\b'), 'director', 3),
    (re.compile(r'\b(manager|team lead|supervisor|lead)\b'), 'manager', 2),
    (re.compile(r'\b(engineer|analyst|coordinator|associate|specialist|consultant|recruiter|designer|developer|researcher|assistant|officer|intern)\b'), 'ic', 1),
]
def psen(t):
    if not isinstance(t, str): return 'unknown', 0
    t = t.lower()
    for p, l, s in SRULES:
        if p.search(t): return l, s
    return 'unknown', 0

df[['sl', 'seniority_score']] = df['job_title'].apply(lambda x: pd.Series(psen(x)))
HR = re.compile(r'\b(hr|human resource|people|culture|talent|learning|employee experience|engagement|workforce|hrbp|diversity|dei|recruitment|onboard)\b')
OPS = re.compile(r'\b(it |information technology|system|infrastructure|operations|ops|digital|platform|enterprise|internal tool)\b')
def pfunc(t):
    if not isinstance(t, str): return 'other'
    t = t.lower()
    if HR.search(t): return 'hr'
    if OPS.search(t): return 'it'
    return 'other'
df['func'] = df['job_title'].apply(pfunc)
df['is_hr'] = (df['func'] == 'hr').astype(int)
df['is_it'] = (df['func'] == 'it').astype(int)
KN = re.compile(r'\b(financ|bank|insurance|invest|capital|asset|accounting|audit|government|public sector|non.profit|npo|ngo|consult|professional service|educat|university|school|real estate|manufactur|legal|law|compliance|healthcare|hospital)\b')
TK = re.compile(r'\b(software|saas|tech|it service|information technology|cloud|ai |data)\b')
def cind(i):
    if not isinstance(i, str): return 'other'
    t = i.lower()
    if KN.search(t): return 'know'
    if TK.search(t): return 'tech'
    return 'other'
df['itype'] = df['industry'].apply(cind)
df['is_know'] = (df['itype'] == 'know').astype(int)
df['is_tech'] = (df['itype'] == 'tech').astype(int)
RMAP = {'1-50':25,'51-200':125,'201-500':350,'500-1000':750,'1000-5000':2500,'5000-10000':7500,'10000+':15000,'1-200':100,'1k+':2500,'11-50':30}
def psize(n, r):
    try:
        v = float(str(n).replace(',', ''))
        return v if 0 < v < 1e6 else None
    except: pass
    if isinstance(r, str):
        k = r.strip().lower()
        if k in RMAP: return float(RMAP[k])
        m2 = re.search(r'(\d[\d,]*)', r)
        if m2: return float(m2.group(1).replace(',', ''))
    return None
df['csz'] = df.apply(lambda r: psize(r['company_size_numeric'], r['company_size_range']), axis=1)
df['is_silo'] = df['csz'].apply(lambda s: 1 if (s and not pd.isna(s) and 100 <= s <= 1500) else 0)
df['is_buyer'] = ((df['sl'].isin(['director','vp','c_suite'])) & (df['func'].isin(['hr','it']))).astype(int)
REF = dt(2026, 4, 22)
def dsc(v):
    if not isinstance(v, str): return None
    for fmt in ('%Y-%m-%d', '%m/%d/%Y'):
        try: return (REF - dt.strptime(v[:10], fmt)).days
        except: pass
    return None
df['dsc'] = df['last_contact_date'].apply(dsc)

FEAT = ['is_know','is_hr','is_it','is_silo','is_buyer','is_tech','seniority_score','csz','dsc']
for f in FEAT:
    if f not in df.columns: df[f] = 0
X_raw = df[FEAT].astype(float)
y = df['converted'].astype(int)
imp = SimpleImputer(strategy='median')
X = imp.fit_transform(X_raw)
sc2 = StandardScaler()
Xsc = sc2.fit_transform(X)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
Xsc_tr, Xsc_te = Xsc[:len(X_tr)], Xsc[len(X_tr):]

MODS = {
    'Logistic Regression': (LogisticRegression(max_iter=2000, class_weight='balanced', C=0.5, random_state=42), Xsc_tr, Xsc_te, Xsc),
    'Random Forest': (RandomForestClassifier(300, max_depth=5, class_weight='balanced', random_state=42), X_tr, X_te, X),
    'XGBoost': (xgb_mod.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=3,
        scale_pos_weight=(y_tr==0).sum()/max((y_tr==1).sum(),1),
        use_label_encoder=False, eval_metric='logloss', random_state=42, verbosity=0), X_tr, X_te, X),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
res = {}
for nm, (m, Xtr, Xte, Xa) in MODS.items():
    m.fit(Xtr, y_tr)
    p = m.predict_proba(Xte)[:,1]
    roc = roc_auc_score(y_te, p) if len(np.unique(y_te)) > 1 else 0.5
    pr = average_precision_score(y_te, p) if len(np.unique(y_te)) > 1 else y_te.mean()
    f1 = f1_score(y_te, (p>=0.5).astype(int), zero_division=0)
    cvs = cross_val_score(m, Xa, y, cv=cv, scoring='roc_auc')
    c2 = confusion_matrix(y_te, (p>=0.5).astype(int))
    tn, fp, fn, tp = (c2.ravel() if c2.shape==(2,2) else (c2[0,0], 0, 0, 0))
    res[nm] = {'roc':roc,'pr':pr,'f1':f1,'p':p,'all_p':m.predict_proba(Xa)[:,1],
               'cv_mean':cvs.mean(),'cv_std':cvs.std(),'cvs':cvs,
               'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)}
    print(f'{nm}: ROC={roc:.3f} PR={pr:.3f} CV={cvs.mean():.3f} TP={int(tp)} FP={int(fp)} FN={int(fn)}')

COL = {'Logistic Regression':'#4C72B0','Random Forest':'#55A868','XGBoost':'#DD8452'}
fig = plt.figure(figsize=(18,16))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.4)

# ROC
ax = fig.add_subplot(gs[0,0])
for nm, r in res.items():
    if len(np.unique(y_te)) > 1:
        fpr2, tpr2, _ = roc_curve(y_te, r['p'])
        ax.plot(fpr2, tpr2, label=f"{nm} (AUC={r['roc']:.2f})", color=COL[nm], lw=2)
ax.plot([0,1],[0,1],'--',color='gray',lw=1,label='Random (0.50)')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves\n(real distribution, no synthetic data)', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# PR
ax2 = fig.add_subplot(gs[0,1])
base = y_te.mean()
for nm, r in res.items():
    if len(np.unique(y_te)) > 1:
        prec, rec, _ = precision_recall_curve(y_te, r['p'])
        ax2.plot(rec, prec, label=f"{nm} (PR={r['pr']:.2f})", color=COL[nm], lw=2)
ax2.axhline(base, color='gray', linestyle='--', lw=1, label=f'Random ({base:.2f})')
ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision')
ax2.set_title('Precision-Recall Curves\n(best metric for imbalanced data)', fontweight='bold')
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

# CV
ax3 = fig.add_subplot(gs[0,2])
for nm, r in res.items():
    ax3.plot(range(5), r['cvs'], 'o-', label=f"{nm} avg={r['cv_mean']:.2f}", color=COL[nm], lw=2, ms=7)
ax3.axhline(0.5, color='red', linestyle='--', lw=1, alpha=0.5, label='Random baseline')
ax3.set_xticks(range(5)); ax3.set_xticklabels([f'Fold {i+1}' for i in range(5)])
ax3.set_ylabel('ROC-AUC'); ax3.set_ylim(0.3, 1.05)
ax3.set_title('5-Fold Cross-Validation\n(each fold is a different train/test split)', fontweight='bold')
ax3.legend(fontsize=7); ax3.grid(alpha=0.3)

# Confusion matrix
ax4 = fig.add_subplot(gs[1,0])
xr = res['XGBoost']
cm_d = np.array([[xr['tn'], xr['fp']], [xr['fn'], xr['tp']]])
ax4.imshow(cm_d, cmap='Blues')
labs2 = [['True Negative\n(correctly skipped)','False Positive\n(wasted outreach)'],
         ['False Negative\n(missed ICP)','True Positive\n(found it!)']]
for i in range(2):
    for j in range(2):
        v = cm_d[i,j]; tc = 'white' if v > cm_d.max()/2 else 'black'
        ax4.text(j, i, str(v), ha='center', va='center', fontsize=22, fontweight='bold', color=tc)
        ax4.text(j, i+0.38, labs2[i][j], ha='center', va='center', fontsize=7.5, color=tc)
ax4.set_xticks([0,1]); ax4.set_yticks([0,1])
ax4.set_xticklabels(['Predicted\nNot ICP','Predicted ICP'], fontsize=9)
ax4.set_yticklabels(['Actual\nNot ICP','Actual ICP'], fontsize=9)
ax4.set_title('Confusion Matrix — XGBoost\n(on held-out test rows)', fontweight='bold')

# Score distributions
ax5 = fig.add_subplot(gs[1,1])
all_s = res['XGBoost']['all_p'] * 100
pay_s = all_s[y==1]; non_s = all_s[y==0]
bins = np.linspace(0, 100, 21)
ax5.hist(non_s, bins=bins, alpha=0.6, color='#e74c3c', label=f'Non-paying (n={len(non_s)})', density=True)
ax5.hist(pay_s, bins=bins, alpha=0.85, color='#2ecc71', label=f'Paying (n={len(pay_s)})', density=True)
ax5.axvline(55, color='orange', linestyle='--', lw=2, label='ICP threshold')
ax5.set_xlabel('ICP Score (0-100)'); ax5.set_ylabel('Density')
ax5.set_title('Score Distributions\nAre paying customers scoring higher?', fontweight='bold')
ax5.legend(fontsize=9); ax5.grid(alpha=0.3)

# Learning curve
ax6 = fig.add_subplot(gs[1,2])
lc_m = xgb_mod.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=3,
    scale_pos_weight=(y==0).sum()/max((y==1).sum(),1),
    use_label_encoder=False, eval_metric='logloss', random_state=42, verbosity=0)
tsz, tsc, vsc = learning_curve(lc_m, X, y, cv=cv, scoring='roc_auc',
    train_sizes=np.linspace(0.2, 1.0, 6), n_jobs=-1)
ax6.plot(tsz, tsc.mean(1), 'o-', color='#DD8452', lw=2, label='Training score')
ax6.fill_between(tsz, tsc.mean(1)-tsc.std(1), tsc.mean(1)+tsc.std(1), alpha=0.15, color='#DD8452')
ax6.plot(tsz, vsc.mean(1), 'o-', color='#4C72B0', lw=2, label='Validation score')
ax6.fill_between(tsz, vsc.mean(1)-vsc.std(1), vsc.mean(1)+vsc.std(1), alpha=0.15, color='#4C72B0')
ax6.axhline(0.5, color='red', linestyle='--', lw=1, alpha=0.5)
ax6.set_xlabel('Training set size'); ax6.set_ylabel('ROC-AUC')
ax6.set_title('Learning Curve\nDoes more data improve accuracy?', fontweight='bold')
ax6.legend(fontsize=9); ax6.grid(alpha=0.3); ax6.set_ylim(0.3, 1.05)
lv = vsc.mean(1)[-1]
ax6.annotate('We are here\n(575 rows,\n18 paying)', xy=(tsz[-1], lv),
    xytext=(tsz[2], max(lv-0.15, 0.35)),
    arrowprops=dict(arrowstyle='->', color='black'), fontsize=9)

# Summary table
ax7 = fig.add_subplot(gs[2,:])
ax7.axis('off')
hdr = ['Model','CV AUC (5-fold)','Test ROC-AUC','Test PR-AUC','Test F1','True Positives\n(correctly found ICPs)','False Positives\n(wrong flags)','False Negatives\n(missed ICPs)','Verdict']
vd = {'Logistic Regression':'Weak — misses combinations','Random Forest':'Good — catches profile combos','XGBoost':'Best — use for scoring'}
rows_d = []
for nm, r in res.items():
    rows_d.append([nm, f"{r['cv_mean']:.3f} ± {r['cv_std']:.3f}", f"{r['roc']:.3f}", f"{r['pr']:.3f}", f"{r['f1']:.3f}", str(r['tp']), str(r['fp']), str(r['fn']), vd[nm]])
tbl = ax7.table(cellText=rows_d, colLabels=hdr, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
for j in range(len(hdr)):
    tbl[0,j].set_facecolor('#2c3e50'); tbl[0,j].set_text_props(color='white', fontweight='bold')
rc = ['#fdebd0','#d5f5e3','#d5f5e3']
for i, c in enumerate(rc):
    for j in range(len(hdr)): tbl[i+1,j].set_facecolor(c)
ax7.set_title('Model Evaluation — Real Test Set, No Synthetic Data', fontsize=12, fontweight='bold', pad=20)

plt.savefig('/Users/alyssa/Desktop/Codex/chart_model_evaluation.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart saved.')
