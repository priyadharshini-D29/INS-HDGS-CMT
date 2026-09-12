"""
Classical ML baselines for the engagement task under the identical LOSOCV protocol.
Features: per-channel relative band-power (5 bands, Welch) per epoch -> (C*5) vector.
Models: Logistic Regression, linear SVM, Random Forest.
CPU-only; no GPU contention. Writes results/baselines/classical_baselines.csv.
"""
import numpy as np, glob, re, json
from pathlib import Path
from scipy.signal import welch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (balanced_accuracy_score, roc_auc_score, matthews_corrcoef,
                             f1_score, accuracy_score)

ROOT = Path("/home/nvidia/24PHD1314/Neuma_Model")
FS = 300.0
BANDS = [(1,4),(4,8),(8,13),(13,30),(30,45)]

def band_powers(epoch):                       # epoch: (T, C) -> (C*5,)
    T, C = epoch.shape
    feats = []
    for c in range(C):
        f, psd = welch(epoch[:, c], fs=FS, nperseg=min(256, T))
        tot = np.trapezoid(psd, f) + 1e-10
        feats += [np.trapezoid(psd[(f>=lo)&(f<hi)], f[(f>=lo)&(f<hi)])/tot for lo,hi in BANDS]
    return np.asarray(feats, np.float32)

# ── load all subjects ─────────────────────────────────────────────────────────
subs = {}
for p in sorted(glob.glob(str(ROOT/"NEUMA_PHASE3/S*/output/epochs/eeg_epochs_engagement.npy"))):
    s = re.search(r"S\d+", p).group()
    lab = ROOT/f"NEUMA_PHASE3/{s}/output/engagement/engagement_labels.npy"
    if not lab.exists(): continue
    eeg = np.load(p, allow_pickle=True).astype(np.float32)            # (E,T,C)
    y = np.asarray(np.load(lab, allow_pickle=True)).ravel().astype(int)
    n = min(len(eeg), len(y)); eeg, y = eeg[:n], y[:n]
    if n < 4: continue
    X = np.stack([band_powers(e) for e in eeg])
    subs[s] = (X, y)

# align feature dim (use modal C*5)
dims = [X.shape[1] for X,_ in subs.values()]
D = max(set(dims), key=dims.count)
subs = {s:(X,y) for s,(X,y) in subs.items() if X.shape[1]==D}
all_ids = sorted(subs)
test_ids = [s for s in all_ids if len(np.unique(subs[s][1]))==2]      # exclude single-class as test
print(f"subjects with data: {len(all_ids)} | evaluable test folds (both classes): {len(test_ids)}")

MODELS = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=2000, class_weight="balanced"),
    "SVM (RBF)":           lambda: SVC(kernel="rbf", class_weight="balanced", probability=True),
    "Random Forest":       lambda: RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42),
}

rows = []
for name, mk in MODELS.items():
    yt_all, yp_all, ys_all = [], [], []
    per = []
    for test in test_ids:
        Xtr = np.concatenate([subs[s][0] for s in all_ids if s!=test])
        ytr = np.concatenate([subs[s][1] for s in all_ids if s!=test])
        Xte, yte = subs[test]
        sc = StandardScaler().fit(Xtr)
        clf = mk().fit(sc.transform(Xtr), ytr)
        proba = clf.predict_proba(sc.transform(Xte))[:,1]
        pred = (proba>=0.5).astype(int)
        yt_all += yte.tolist(); yp_all += pred.tolist(); ys_all += proba.tolist()
        try: auc = roc_auc_score(yte, proba) if len(set(yte))>1 else np.nan
        except Exception: auc = np.nan
        per.append((balanced_accuracy_score(yte,pred), auc, matthews_corrcoef(yte,pred)))
    yt=np.array(yt_all); yp=np.array(yp_all); ys=np.array(ys_all)
    per=np.array(per, float)
    r = dict(model=name,
             bal_acc_pf=np.nanmean(per[:,0]), auc_pf=np.nanmean(per[:,1]), mcc_pf=np.nanmean(per[:,2]),
             bal_acc_pool=balanced_accuracy_score(yt,yp), auc_pool=roc_auc_score(yt,ys),
             mcc_pool=matthews_corrcoef(yt,yp), f1_pool=f1_score(yt,yp), acc_pool=accuracy_score(yt,yp))
    rows.append(r)
    print(f"{name:22s} bal={r['bal_acc_pool']:.3f} auc={r['auc_pool']:.3f} "
          f"mcc={r['mcc_pool']:.3f} f1={r['f1_pool']:.3f} acc={r['acc_pool']:.3f} "
          f"(per-fold bal={r['bal_acc_pf']:.3f} auc={r['auc_pf']:.3f})")

out = ROOT/"NEUMA_PHASE8/results/baselines"; out.mkdir(parents=True, exist_ok=True)
import csv
with open(out/"classical_baselines.csv","w",newline="") as f:
    w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); [w.writerow(r) for r in rows]
json.dump({"n_test_folds":len(test_ids),"feature_dim":int(D)}, open(out/"meta.json","w"))
print(f"\nwrote {out}/classical_baselines.csv  (n_test_folds={len(test_ids)}, feat_dim={D})")
