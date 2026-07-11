import numpy as np


def compute_stream_statistics(streams):

    print("\n===== STREAM STATISTICS =====")

    summary = {}

    for name, stream in streams.items():

        ts = np.array(stream["time_stamps"])
        duration = ts[-1] - ts[0]
        effective_sr = (len(ts) - 1) / duration if duration > 0 else 0

        summary[name] = {
            "samples": len(ts),
            "duration_sec": duration,
            "effective_sr": effective_sr,
        }

        print(
            f"{name:15s} | "
            f"samples={len(ts):6d}  "
            f"duration={duration:.2f}s  "
            f"SR={effective_sr:.2f}Hz"
        )

    return summary
