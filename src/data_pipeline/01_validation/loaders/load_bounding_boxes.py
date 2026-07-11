import pandas as pd
from configs.paths import BOUNDING_BOX_DIR


def load_bounding_boxes(filename):
    path = BOUNDING_BOX_DIR / filename
    if not path.exists():
        print(f"[MISSING] Bounding box file not found: {path}")
        return None
    return pd.read_csv(path)
