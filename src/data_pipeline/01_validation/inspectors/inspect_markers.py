import numpy as np


def inspect_markers(marker_stream):
    print("\n===== MARKER STREAM INSPECTION =====")

    series = marker_stream["time_series"]
    timestamps = np.array(marker_stream["time_stamps"])

    print(f"Total Markers      : {len(series)}")
    print(f"Timestamp Start    : {timestamps[0]:.4f}")
    print(f"Timestamp End      : {timestamps[-1]:.4f}")

    print("\nFirst 20 markers:")
    for ts, marker in zip(timestamps[:20], series[:20]):
        label = marker[0] if isinstance(marker, (list, tuple)) else marker
        print(f"  t={ts:.4f}  ->  {label}")
