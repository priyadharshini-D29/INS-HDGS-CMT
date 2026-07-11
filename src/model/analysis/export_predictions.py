import numpy as np
import pandas as pd
from pathlib import Path

# Was a hardcoded absolute path from the original dev box (already broken
# on any other machine); fixed opportunistically to a portable
# file-relative path while touching this file for the src/model/ rename.
_MODEL_DIR = Path(__file__).resolve().parent.parent
ROOT = _MODEL_DIR / "output" / "metrics"

csvs = sorted(ROOT.glob("*.csv"))

print("\n===================================")
print(" EXPORT POOLED PREDICTIONS")
print("===================================")

all_rows = []

for f in csvs:

    try:

        df = pd.read_csv(f)

        print(f"[OK] {f.name}")

        print(df.columns.tolist())

        if "y_true" in df.columns and "y_prob" in df.columns:

            all_rows.append(
                df[["y_true", "y_prob"]]
            )

    except Exception as e:

        print(f"[FAIL] {f.name}: {e}")

if len(all_rows) == 0:

    print("\nNo prediction CSVs found.")
    exit()

final_df = pd.concat(all_rows, ignore_index=True)

save_path = _MODEL_DIR / "output" / "predictions.csv"

final_df.to_csv(save_path, index=False)

print("\n===================================")
print(" COMPLETE")
print("===================================")

print(final_df.head())

print("\nSaved:")
print(save_path)
