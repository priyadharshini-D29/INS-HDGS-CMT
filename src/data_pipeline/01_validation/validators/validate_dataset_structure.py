from configs.paths import (
    XDF_DIR, QUESTIONNAIRE_DIR, BROCHURE_DIR,
    BOUNDING_BOX_DIR, PRODUCT_METADATA_DIR, CHANNEL_LOCATION_DIR
)

REQUIRED_DIRS = [
    XDF_DIR,
    QUESTIONNAIRE_DIR,
    BROCHURE_DIR,
    BOUNDING_BOX_DIR,
    PRODUCT_METADATA_DIR,
    CHANNEL_LOCATION_DIR
]


def validate_structure():
    print("\n===== DATASET STRUCTURE VALIDATION =====")

    for folder in REQUIRED_DIRS:
        if folder.exists():
            print(f"[PASS] {folder}")
        else:
            print(f"[FAIL] Missing: {folder}")
