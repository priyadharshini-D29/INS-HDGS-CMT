import numpy as np

from configs.thresholds import ET_NAN_WARNING_RATIO


def validate_et_signal(et_stream):

    et = np.array(et_stream["time_series"])

    print("\n===== ET SIGNAL VALIDATION =====")

    print(f"Shape : {et.shape}")

    for ch in range(et.shape[1]):

        data = et[:, ch]

        nan_ratio = np.isnan(data).mean()

        valid_ratio = np.mean(~np.isnan(data))

        status = "[WARN]" if nan_ratio > ET_NAN_WARNING_RATIO else "[PASS]"

        print(
            f"{status} Ch{ch} | "
            f"NaN ratio={nan_ratio:.4f} "
            f"Valid ratio={valid_ratio:.4f}"
        )
