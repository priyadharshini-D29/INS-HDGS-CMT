import sys
from pathlib import Path

# Allow running from inside the src/data_pipeline/01_validation directory
sys.path.insert(0, str(Path(__file__).resolve().parent))

from configs.paths import XDF_DIR, BROCHURE_DIR, BOUNDING_BOX_DIR

from validators.validate_dataset_structure import validate_structure
from validators.validate_subject_files import validate_subject_xdf_files
from validators.validate_stream_existence import validate_streams
from validators.validate_sampling_rates import validate_sampling_rates
from validators.validate_basic_integrity import validate_basic_integrity

from loaders.load_xdf import load_xdf_file, identify_streams

from inspectors.inspect_xdf_streams import inspect_streams
from inspectors.inspect_eeg_stream import inspect_eeg_stream
from inspectors.inspect_et_stream import inspect_et_stream
from inspectors.inspect_mouse_stream import inspect_mouse_stream
from inspectors.inspect_markers import inspect_markers
from inspectors.deep_inspect_et_stream import deep_inspect_et_stream
from inspectors.inspect_roi_mapping import inspect_roi_mapping

from visualization.visualize_stream_timeline import plot_stream_timeline


def main():

    print("\n==============================")
    print(" NEUMA PHASE 0 ")
    print(" DATASET UNDERSTANDING ")
    print("==============================")

    validate_structure()
    validate_subject_xdf_files()

    xdf_files = sorted(XDF_DIR.glob("*.xdf"))

    if len(xdf_files) == 0:
        print("\nNo XDF files found.")
        return

    sample_file = xdf_files[0]
    print(f"\nUsing sample file: {sample_file.name}")

    validate_streams(sample_file)
    validate_sampling_rates(sample_file)
    validate_basic_integrity(sample_file)

    inspect_streams(sample_file)

    streams, _ = load_xdf_file(sample_file)
    identified = identify_streams(streams)

    if identified["EEG"] is not None:
        inspect_eeg_stream(identified["EEG"])

    if identified["ET"] is not None:
        #inspect_et_stream(identified["ET"])
        deep_inspect_et_stream(identified["ET"])
    if identified["MousePosition"] is not None:
        inspect_mouse_stream(identified["MousePosition"])

    if identified["Markers"] is not None:
        inspect_markers(identified["Markers"])

    plot_stream_timeline(identified)

    brochure_path = BROCHURE_DIR / "ImagePage_1.tif"

    roi_path = BOUNDING_BOX_DIR / "BoundingBoxPage_1.mat"

    inspect_roi_mapping(
        brochure_path,
        roi_path,
        identified["ET"]
    )

if __name__ == "__main__":
    main()
