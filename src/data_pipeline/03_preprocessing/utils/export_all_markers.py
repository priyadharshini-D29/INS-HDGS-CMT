import os
import glob
import pyxdf
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

# cwd-relative to the phase root (03_preprocessing/), now nested two levels
# deeper than before (src/data_pipeline/03_preprocessing), hence "../../../".
XDF_DIR = r"../../../DataSource"

OUTPUT_DIR = r"output/events"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# FIND ALL XDF FILES
# ============================================================

xdf_files = glob.glob(
    os.path.join(XDF_DIR, "*.xdf")
)

print(f"\nDetected XDF files: {len(xdf_files)}")


# ============================================================
# PROCESS EACH SUBJECT
# ============================================================

for xdf_path in xdf_files:

    subject_id = os.path.basename(
        xdf_path
    ).replace(".xdf", "")

    print("\n================================")
    print(f"Processing {subject_id}")
    print("================================")

    # --------------------------------------------------------
    # LOAD XDF
    # --------------------------------------------------------

    streams, header = pyxdf.load_xdf(
        xdf_path
    )

    marker_stream = None

    # --------------------------------------------------------
    # FIND MARKER STREAM
    # --------------------------------------------------------

    for stream in streams:

        name = stream["info"]["name"][0]

        if "marker" in name.lower():

            marker_stream = stream
            break

    if marker_stream is None:

        print("[WARN] No marker stream found")
        continue

    # --------------------------------------------------------
    # EXTRACT MARKERS
    # --------------------------------------------------------

    timestamps = marker_stream["time_stamps"]

    samples = marker_stream["time_series"]

    rows = []

    for ts, sample in zip(timestamps, samples):

        try:
            label = str(sample[0])

        except:
            label = "UNKNOWN"

        rows.append({
            "timestamp": ts,
            "label": label
        })

    df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_csv = os.path.join(
        OUTPUT_DIR,
        f"{subject_id}_markers.csv"
    )

    df.to_csv(
        output_csv,
        index=False
    )

    print(f"Saved: {output_csv}")

print("\n================================")
print(" MARKER EXPORT COMPLETE ")
print("================================")