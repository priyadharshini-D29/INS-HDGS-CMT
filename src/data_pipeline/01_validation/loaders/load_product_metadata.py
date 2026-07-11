import pandas as pd
from configs.paths import PRODUCT_METADATA_DIR


def load_product_metadata(filename):
    path = PRODUCT_METADATA_DIR / filename
    if not path.exists():
        print(f"[MISSING] Product metadata file not found: {path}")
        return None
    return pd.read_csv(path)
