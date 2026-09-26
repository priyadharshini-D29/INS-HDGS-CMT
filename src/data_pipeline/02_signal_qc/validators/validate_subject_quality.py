import numpy as np

from configs.thresholds import ET_NAN_WARNING_RATIO, SUBJECT_MIN_DURATION_SEC


def compute_subject_quality(streams):

    score = 100
    reasons = []

    eeg = np.array(streams["EEG"]["time_series"])
    et = np.array(streams["ET"]["time_series"])

    eeg_ts = np.array(streams["EEG"]["time_stamps"])
    duration = eeg_ts[-1] - eeg_ts[0]

    eeg_nan = np.isnan(eeg).sum()
    et_nan_ratio = np.isnan(et).mean()

    if eeg_nan > 0:
        score -= 20
        reasons.append(f"EEG NaN count={eeg_nan}")

    if et_nan_ratio > ET_NAN_WARNING_RATIO:
        score -= 20
        reasons.append(f"ET NaN ratio={et_nan_ratio:.2f}")

    if duration < SUBJECT_MIN_DURATION_SEC:
        score -= 20
        reasons.append(f"Short duration={duration:.1f}s")

    print("\n===== SUBJECT QUALITY SCORE =====")
    print(f"Duration (sec)  : {duration:.1f}")
    print(f"Quality Score   : {score}/100")

    if reasons:
        for r in reasons:
            print(f"  [-] {r}")

    if score >= 90:
        print("Grade: Excellent")
    elif score >= 70:
        print("Grade: Usable")
    else:
        print("Grade: Poor")
