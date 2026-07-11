import numpy as np
from pathlib import Path


def load_clean_data(subject_id: str = None):
    """
    Load Phase-2 cleaned EEG, ET, and timestamps for one subject.

    Parameters
    ----------
    subject_id : canonical string e.g. "S01".
                 Defaults to NEUMA_SUBJECT env var (set by main_phase3.py
                 via argparse before any project imports).
    """
    from config.settings import get_paths, _SUBJECT_ID
    sid = subject_id or _SUBJECT_ID
    p   = get_paths(sid)

    print(f"\n===== LOADING CLEAN DATA ({sid}) =====")
    print(f"  Phase 2 dir : {p.phase2_dir}")

    # Fail fast with a helpful message if Phase 2 hasn't been run yet
    missing = [
        str(path) for path in (p.eeg_clean, p.et_clean,
                                p.eeg_timestamps, p.et_timestamps)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"\n[Phase 3] Missing Phase 2 outputs for subject {sid!r}:\n"
            + "\n".join(f"  {m}" for m in missing)
            + f"\n\nFix: python src/data_pipeline/03_preprocessing/main_phase2.py --subject {sid}"
        )

    eeg    = np.load(p.eeg_clean)
    et     = np.load(p.et_clean)
    eeg_ts = np.load(p.eeg_timestamps)
    et_ts  = np.load(p.et_timestamps)

    # Resolve timestamp length mismatch between et_clean and et_timestamps
    if len(et_ts) != len(et):
        print(f"[WARN] et_timestamps length ({len(et_ts)}) != et rows ({len(et)})")
        orig = p.phase2_dir / "et_timestamps_original.npy"
        if orig.exists():
            et_ts_orig = np.load(orig)
            if len(et_ts_orig) == len(et):
                print(f"       Using et_timestamps_original.npy ({len(et_ts_orig)} entries)")
                et_ts = et_ts_orig
            else:
                print("       Falling back to eeg_timestamps")
                et_ts = eeg_ts
        else:
            print("       et_timestamps_original.npy not found — using eeg_timestamps")
            et_ts = eeg_ts

    print(f"EEG : {eeg.shape}  ({eeg_ts[0]:.3f} -> {eeg_ts[-1]:.3f} s)")
    print(f"ET  : {et.shape}   ({et_ts[0]:.3f} -> {et_ts[-1]:.3f} s)")
    return eeg, et, eeg_ts, et_ts
