import pandas as pd
from configs.paths import CHANNEL_LOCATION_DIR


def load_channel_locations(filename):
    path = CHANNEL_LOCATION_DIR / filename
    if not path.exists():
        print(f"[MISSING] Channel locations file not found: {path}")
        return None
    return pd.read_csv(path)
