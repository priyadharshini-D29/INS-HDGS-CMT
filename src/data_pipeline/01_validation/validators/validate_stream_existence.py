from loaders.load_xdf import load_xdf_file, identify_streams


def validate_streams(xdf_path):
    print(f"\n===== STREAM VALIDATION: {xdf_path.name} =====")

    streams, _ = load_xdf_file(xdf_path)
    identified = identify_streams(streams)

    for key, value in identified.items():
        if value is not None:
            print(f"[PASS] {key} stream exists")
        else:
            print(f"[FAIL] {key} stream missing")
