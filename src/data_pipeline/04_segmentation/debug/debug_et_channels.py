import numpy as np

et = np.load(
    "../03_preprocessing/output/clean_data/et_clean.npy"
)

print("\n===== ET CHANNEL DEBUG =====")

print(f"Shape : {et.shape}")

for ch in range(et.shape[1]):

    data = et[:, ch]

    print(f"\nChannel {ch}")

    print(f"Min  : {np.nanmin(data):.6f}")
    print(f"Max  : {np.nanmax(data):.6f}")
    print(f"Mean : {np.nanmean(data):.6f}")
    print(f"Std  : {np.nanstd(data):.6f}")

    print("First 10 values:")
    print(data[:10])