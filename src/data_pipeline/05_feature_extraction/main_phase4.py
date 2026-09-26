import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Universal UTF-8 / Unicode-safe I/O ────────────────────────────────────────
import sys as _sys, os as _os, io as _io
_os.environ.setdefault("PYTHONUTF8", "1")
try:
    if isinstance(_sys.stdout, _io.TextIOWrapper):
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if isinstance(_sys.stderr, _io.TextIOWrapper):
        _sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_REPL = {"→":"->","←":"<-","✔":"[OK]","✘":"[X]","─":"-","│":"|","═":"=",
         "┌":"+","┐":"+","└":"+","┘":"+","├":"+","┤":"+","█":"#","░":".","►":">","±":"+/-","∞":"inf"}

def sanitize_text(t):
    t = str(t)
    for o, n in _REPL.items(): t = t.replace(o, n)
    return t

def safe_print(*args, **kwargs):
    sep_c = kwargs.pop("sep", " ")
    text  = sanitize_text(sep_c.join(str(a) for a in args))
    try:    print(text, **kwargs)
    except UnicodeEncodeError: print(text.encode("ascii","ignore").decode(), **kwargs)
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd

from config.settings import OUTPUT_FEATURES, OUTPUT_PLOTS

from loaders.load_epochs import load_epochs

from eeg.eeg_features import extract_eeg_features, feature_names as eeg_feature_names
from et.et_features    import extract_et_features,  et_feature_names
from fusion.fusion_features import build_feature_matrix

from visualization.plot_feature_distributions import (
    plot_feature_distributions,
    plot_feature_heatmap
)

for d in [OUTPUT_FEATURES, OUTPUT_PLOTS]:
    Path(d).mkdir(parents=True, exist_ok=True)


def main():

    print("\n==============================")
    print(" PHASE 4 → FEATURE ENGINEERING")
    print("==============================")

    eeg_epochs, et_epochs, labels = load_epochs()

    n = len(eeg_epochs)
    eeg_feature_list = []
    et_feature_list  = []

    # ── Extract features ─────────────────────────────────────
    for i in range(n):

        eeg_feat = extract_eeg_features(eeg_epochs[i])
        et_feat  = extract_et_features(et_epochs[i])

        eeg_feature_list.append(eeg_feat)
        et_feature_list.append(et_feat)

        print(f"  [{i+1:3d}/{n}]  EEG={eeg_feat.shape[0]}  ET={et_feat.shape[0]}")

    # ── Build matrices ───────────────────────────────────────
    eeg_features, et_features, fusion_features = build_feature_matrix(
        eeg_feature_list, et_feature_list
    )

    # ── Feature names ────────────────────────────────────────
    n_ch    = eeg_epochs[0].shape[1]
    eeg_names    = eeg_feature_names(n_ch)
    et_names     = et_feature_names()
    fusion_names = eeg_names + et_names

    # ── Save ─────────────────────────────────────────────────
    np.save(OUTPUT_FEATURES / "eeg_features.npy",    eeg_features)
    np.save(OUTPUT_FEATURES / "et_features.npy",     et_features)
    np.save(OUTPUT_FEATURES / "fusion_features.npy", fusion_features)
    np.save(OUTPUT_FEATURES / "labels.npy",          labels)

    pd.DataFrame(eeg_features,    columns=eeg_names).to_csv(
        OUTPUT_FEATURES / "eeg_features.csv",    index=False)
    pd.DataFrame(et_features,     columns=et_names).to_csv(
        OUTPUT_FEATURES / "et_features.csv",     index=False)
    pd.DataFrame(fusion_features, columns=fusion_names).to_csv(
        OUTPUT_FEATURES / "fusion_features.csv", index=False)

    print("\n===== FEATURE SUMMARY =====")
    print(f"EEG Features    : {eeg_features.shape}")
    print(f"ET  Features    : {et_features.shape}")
    print(f"Fusion Features : {fusion_features.shape}")
    print(f"Labels          : {labels.shape}")

    # ── Visualisations ───────────────────────────────────────
    print("\nGenerating plots...")

    plot_feature_distributions(
        eeg_features, eeg_names, n_show=20,
        save_path=OUTPUT_PLOTS / "eeg_feature_distributions.png"
    )

    plot_feature_distributions(
        et_features, et_names, n_show=6,
        save_path=OUTPUT_PLOTS / "et_feature_distributions.png"
    )

    plot_feature_heatmap(
        fusion_features, labels,
        save_path=OUTPUT_PLOTS / "fusion_feature_heatmap.png"
    )

    print("\nSaved:")
    print(f"  {OUTPUT_FEATURES}/eeg_features.npy  + .csv")
    print(f"  {OUTPUT_FEATURES}/et_features.npy   + .csv")
    print(f"  {OUTPUT_FEATURES}/fusion_features.npy + .csv")
    print(f"  {OUTPUT_FEATURES}/labels.npy")
    print(f"  {OUTPUT_PLOTS}/eeg_feature_distributions.png")
    print(f"  {OUTPUT_PLOTS}/et_feature_distributions.png")
    print(f"  {OUTPUT_PLOTS}/fusion_feature_heatmap.png")

    print("\n==============================")
    print(" PHASE 4 COMPLETE")
    print("==============================")


if __name__ == "__main__":
    main()
