from pathlib import Path


def list_xdf_files(xdf_dir):
    return sorted(Path(xdf_dir).glob("*.xdf"))


def subject_id_from_path(xdf_path):
    return Path(xdf_path).stem
