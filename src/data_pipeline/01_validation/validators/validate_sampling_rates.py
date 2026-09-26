import numpy as np
from configs.constants import EXPECTED_EEG_SR, EXPECTED_ET_SR
from loaders.load_xdf import load_xdf_file, identify_streams


def validate_sampling_rates(xdf_path):
    print(f"\n===== SAMPLING RATE VALIDATION: {xdf_path.name} =====")

    streams, _ = load_xdf_file(xdf_path)
    identified = identify_streams(streams)

    if identified["EEG"] is not None:
        sr = float(identified["EEG"]["info"].get("nominal_srate", [0])[0])
        status = "[PASS]" if sr == EXPECTED_EEG_SR else "[WARN]"
        print(f"{status} EEG sampling rate: {sr} Hz (expected {EXPECTED_EEG_SR})")
    else:
        print("[FAIL] EEG stream not found for SR check")

    if identified["ET"] is not None:
        sr = float(identified["ET"]["info"].get("nominal_srate", [0])[0])
        status = "[PASS]" if sr == EXPECTED_ET_SR else "[WARN]"
        print(f"{status} ET sampling rate: {sr} Hz (expected {EXPECTED_ET_SR})")
    else:
        print("[FAIL] ET stream not found for SR check")
