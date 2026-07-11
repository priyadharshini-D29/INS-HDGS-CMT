import numpy as np
import pandas as pd


def extract_events(markers_df):
    """
    Parse a markers DataFrame into a list of event dicts.

    Returns list of {"timestamp": float, "label": str}.
    Rows with unparseable timestamps are silently skipped.
    """
    events = []

    for _, row in markers_df.iterrows():
        try:
            events.append({
                "timestamp": float(row["timestamp"]),
                "label":     str(row["label"])
            })
        except (ValueError, KeyError):
            continue

    print(f"\n===== EVENTS =====")
    print(f"Valid events : {len(events)}")

    label_counts = pd.Series([e["label"] for e in events]).value_counts()
    print(label_counts.to_string())

    return events


def filter_events_by_label(events, keep_labels):
    """Return only events whose label is in keep_labels."""
    filtered = [e for e in events if e["label"] in keep_labels]
    print(f"Events after label filter ({keep_labels}): {len(filtered)}")
    return filtered
