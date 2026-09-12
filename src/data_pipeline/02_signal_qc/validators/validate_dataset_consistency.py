from pathlib import Path
import pyxdf


def validate_dataset_consistency(xdf_dir):

    print("\n===== DATASET CONSISTENCY =====")

    xdf_files = sorted(Path(xdf_dir).glob("*.xdf"))

    print(f"Total subjects found: {len(xdf_files)}")

    for xdf_path in xdf_files:

        streams, _ = pyxdf.load_xdf(str(xdf_path))

        stream_names = [s["info"]["name"][0] for s in streams]

        has_eeg = any("EEG" in n or "WS" in n for n in stream_names)
        has_et = any("Tobii" in n for n in stream_names)
        has_markers = any("Marker" in n for n in stream_names)

        status = "[PASS]" if (has_eeg and has_et and has_markers) else "[WARN]"

        print(
            f"{status} {xdf_path.name:12s} | "
            f"EEG={has_eeg} ET={has_et} Markers={has_markers}"
        )
