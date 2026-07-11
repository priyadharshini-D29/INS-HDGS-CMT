from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

# Actual raw data lives in DataSource at the repo root. ROOT_DIR is now
# nested two levels deeper than before (src/data_pipeline/01_validation
# instead of NEUMA_PHASE0 directly under the repo root), so three .parent
# hops are needed instead of one.
DATA_SOURCE = ROOT_DIR.parent.parent.parent / "DataSource"

RAW_DATA = ROOT_DIR / "data" / "raw"

# XDF and questionnaire files are flat in DataSource
XDF_DIR = DATA_SOURCE
QUESTIONNAIRE_DIR = DATA_SOURCE

DEPENDENCIES_DIR = DATA_SOURCE / "Dependencies"

BROCHURE_DIR = DEPENDENCIES_DIR / "Brochure_Pages"
BOUNDING_BOX_DIR = DEPENDENCIES_DIR / "BoundingBox_Coordinates"
PRODUCT_METADATA_DIR = DEPENDENCIES_DIR
CHANNEL_LOCATION_DIR = DEPENDENCIES_DIR

CHANLOCS_FILE = DEPENDENCIES_DIR / "chanlocs_dsi24.mat"
LEAFLET_FILE = DEPENDENCIES_DIR / "Leaflet_Product_Descriptions.mat"

REPORT_DIR = ROOT_DIR / "reports"
LOG_DIR = ROOT_DIR / "logs"
