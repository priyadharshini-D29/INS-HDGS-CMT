import numpy as np


def validate_stream_alignment(streams):

    print("\n===== STREAM ALIGNMENT =====")

    starts = {}
    ends = {}

    for name, stream in streams.items():

        ts = stream["time_stamps"]

        starts[name] = ts[0]
        ends[name] = ts[-1]

        print(
            f"{name:15s} | "
            f"start={ts[0]:.3f} "
            f"end={ts[-1]:.3f} "
            f"duration={ts[-1]-ts[0]:.2f}s"
        )

    overall_start = max(starts.values())
    overall_end = min(ends.values())
    overlap = overall_end - overall_start

    print(f"\nOverlapping window  : {overlap:.2f}s")

    if overlap > 0:
        print("[PASS] All streams overlap")
    else:
        print("[FAIL] Streams do not fully overlap")
