import numpy as np
import pandas as pd
from pathlib import Path


def load_markers_csv(path=None):
    """
    Load pre-exported markers.csv (timestamp, label).

    path : explicit Path or str. If None, uses settings.MARKERS_CSV
           which is resolved from the current NEUMA_SUBJECT env var.
    """
    if path is not None:
        csv_path = Path(path)
    else:
        from config.settings import MARKERS_CSV
        csv_path = MARKERS_CSV

    print("\n===== LOADING MARKERS =====")
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Markers file not found: {csv_path}\n"
            "Run utils/export_markers.py first to extract from XDF."
        )

    df = pd.read_csv(csv_path)
    print(f"Marker rows : {len(df)}")
    print(df.head())
    return df


def load_markers_from_stream(marker_stream):
    """
    Parse markers directly from a loaded XDF marker stream.
    Returns a DataFrame with columns: timestamp, label.
    """
    timestamps = np.array(marker_stream["time_stamps"])
    series     = marker_stream["time_series"]

    records = []
    for ts, marker in zip(timestamps, series):
        label = marker[0] if isinstance(marker, (list, tuple)) else str(marker)
        records.append({"timestamp": ts, "label": label})

    df = pd.DataFrame(records)
    print(f"\n===== MARKERS FROM STREAM =====")
    print(f"Total markers : {len(df)}")
    print(df.head(10))
    return df
