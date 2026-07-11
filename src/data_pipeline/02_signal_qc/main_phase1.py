import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from loaders.load_xdf import load_xdf_file

from validators.validate_timestamps import validate_timestamps
from validators.validate_eeg_signal import validate_eeg_signal
from validators.validate_et_signal import validate_et_signal
from validators.validate_stream_alignment import validate_stream_alignment
from validators.validate_subject_quality import compute_subject_quality

from validators.validate_frequency_domain import validate_frequency_domain
from validators.validate_channel_correlation import validate_channel_correlation
from validators.validate_artifacts import validate_artifacts

from analyzers.eeg_statistics import compute_eeg_statistics
from analyzers.et_statistics import compute_et_statistics
from analyzers.stream_statistics import compute_stream_statistics

from visualization.plot_eeg_quality import plot_eeg_quality
from visualization.plot_et_quality import plot_et_quality
from visualization.plot_timestamp_drift import plot_timestamp_drift
from visualization.plot_psd import plot_psd
from visualization.plot_channel_heatmap import plot_channel_heatmap
from visualization.plot_blink_events import plot_blink_events



# cwd-relative to the phase root (02_signal_qc/), now nested two levels
# deeper than before (src/data_pipeline/02_signal_qc), hence "../../../".
XDF_PATH = r"../../../DataSource/S01.xdf"


def main():

    print("\n==============================")
    print(" PHASE 1 → RAW STREAM VALIDATION ")
    print("==============================")

    streams = load_xdf_file(XDF_PATH)

    # -----------------------------------
    # Timestamp validation
    # -----------------------------------

    for name, stream in streams.items():
        validate_timestamps(stream, name)

    # -----------------------------------
    # EEG validation
    # -----------------------------------

    validate_eeg_signal(streams["EEG"])

    # -----------------------------------
    # ET validation
    # -----------------------------------

    validate_et_signal(streams["ET"])

    # -----------------------------------
    # Stream synchronization
    # -----------------------------------

    validate_stream_alignment(streams)

    # -----------------------------------
    # Subject quality
    # -----------------------------------

    compute_subject_quality(streams)

    # -----------------------------------
    # Statistics
    # -----------------------------------

    compute_stream_statistics(streams)
    compute_eeg_statistics(streams["EEG"])
    compute_et_statistics(streams["ET"])

    # -----------------------------------
    # Visualizations
    # -----------------------------------

    plot_eeg_quality(streams["EEG"], subject_id="S01")
    plot_et_quality(streams["ET"], subject_id="S01")
    plot_timestamp_drift(streams)
    
    
    # -----------------------------------
    # Scientific signal validation
    # -----------------------------------

    validate_frequency_domain(streams["EEG"])

    corr = validate_channel_correlation(streams["EEG"])

    validate_artifacts(
        streams["EEG"],
        streams["ET"]
    )

    # -----------------------------------
    # Visualizations
    # -----------------------------------

    plot_psd(streams["EEG"])

    plot_channel_heatmap(corr)

    plot_blink_events(streams["ET"])


if __name__ == "__main__":
    main()
