import numpy as np
from loaders.load_xdf import load_xdf_file, identify_streams


def validate_basic_integrity(xdf_path):
    print(f"\n===== BASIC INTEGRITY VALIDATION: {xdf_path.name} =====")

    streams, _ = load_xdf_file(xdf_path)
    identified = identify_streams(streams)

    for name, stream in identified.items():
        if stream is None:
            print(f"[SKIP] {name}: not present")
            continue

        ts = np.array(stream["time_stamps"])
        data = np.array(stream["time_series"])

        if len(ts) == 0:
            print(f"[FAIL] {name}: empty timestamps")
            continue

        diffs = np.diff(ts)
        backwards = (diffs < 0).sum()

        print(f"[INFO] {name}: {len(ts)} samples, "
              f"duration={ts[-1]-ts[0]:.2f}s, "
              f"non-monotonic jumps={backwards}")
