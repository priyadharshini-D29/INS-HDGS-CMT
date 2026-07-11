"""
Extract the marker stream from an XDF file and save as markers.csv.
Called automatically by main_phase3.py when markers.csv is missing.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import pyxdf

from config.settings import MARKERS_CSV, XDF_DIR


def extract_markers_from_xdf(xdf_path, out_path=MARKERS_CSV):

    print(f"\n===== EXTRACTING MARKERS =====")
    print(f"Source : {xdf_path}")

    streams, _ = pyxdf.load_xdf(str(xdf_path))

    marker_stream = None
    for stream in streams:
        name = stream["info"]["name"][0]
        if "Marker" in name or "marker" in name:
            marker_stream = stream
            print(f"Found marker stream: {name}")
            break

    if marker_stream is None:
        raise RuntimeError("No marker stream found in XDF file.")

    timestamps = np.array(marker_stream["time_stamps"])
    series     = marker_stream["time_series"]

    records = []
    for ts, marker in zip(timestamps, series):
        label = marker[0] if isinstance(marker, (list, tuple)) else str(marker)
        records.append({"timestamp": float(ts), "label": str(label)})

    df = pd.DataFrame(records)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Saved  : {out_path}  ({len(df)} markers)")
    print(df.head(10).to_string(index=False))
    print(f"\nUnique labels:\n{df['label'].value_counts().to_string()}")

    return df


if __name__ == "__main__":
    subject = sys.argv[1] if len(sys.argv) > 1 else "S01"
    xdf_path = XDF_DIR / f"{subject}.xdf"
    extract_markers_from_xdf(xdf_path)
