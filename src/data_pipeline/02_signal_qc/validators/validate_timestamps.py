import numpy as np

from configs.thresholds import TIMESTAMP_GAP_WARNING_SEC


def validate_timestamps(stream, stream_name):

    timestamps = np.array(stream["time_stamps"])

    diffs = np.diff(timestamps)

    negative_jumps = np.sum(diffs <= 0)

    max_gap = np.max(diffs)

    print(f"\n===== {stream_name} TIMESTAMP VALIDATION =====")

    print(f"Samples             : {len(timestamps)}")
    print(f"Negative jumps      : {negative_jumps}")
    print(f"Maximum gap (sec)   : {max_gap:.6f}")

    if negative_jumps == 0:
        print("[PASS] Monotonic timestamps")
    else:
        print("[FAIL] Non-monotonic timestamps")

    if max_gap > TIMESTAMP_GAP_WARNING_SEC:
        print(f"[WARN] Large gap detected: {max_gap:.4f}s > {TIMESTAMP_GAP_WARNING_SEC}s threshold")
