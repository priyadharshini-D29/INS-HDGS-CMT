from configs.paths import XDF_DIR


def validate_subject_xdf_files():
    print("\n===== SUBJECT FILE VALIDATION =====")

    xdf_files = sorted(XDF_DIR.glob("*.xdf"))

    print(f"Total XDF Files Found: {len(xdf_files)}")

    for file in xdf_files:
        print(f"[FOUND] {file.name}")
