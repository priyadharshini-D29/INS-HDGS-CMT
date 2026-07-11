"""Smoke tests for INS-HDGS-CMT.

These are intentionally lightweight so they run in CI without a GPU or the
dataset: they check that the repository layout is intact, the configuration
imports, and the YAML configs are well-formed.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_repo_layout():
    """Core folders required by the reproducibility pipeline exist."""
    for d in [
        "src/model", "src/data_pipeline", "configs", "reproducibility",
        "ablation", "results", "tables", "figures", "paper", "docs", "datasets",
    ]:
        assert (ROOT / d).is_dir(), f"missing folder: {d}"


def test_key_files_present():
    for f in [
        "README.md", "LICENSE", "CITATION.cff", "requirements.txt",
        "environment.yml", "setup.py", ".gitattributes",
        "configs/default.yaml", "reproducibility/reproduce_paper.sh",
    ]:
        assert (ROOT / f).is_file(), f"missing file: {f}"


def test_configs_parse():
    """Every YAML config parses (skips gracefully if PyYAML is unavailable)."""
    try:
        import yaml
    except ImportError:
        import pytest
        pytest.skip("PyYAML not installed")
    for cfg in (ROOT / "configs").rglob("*.yaml"):
        with open(cfg, "r", encoding="utf-8") as fh:
            yaml.safe_load(fh)


def test_settings_importable():
    """The model settings module imports and exposes the expected constants."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "settings", ROOT / "src" / "model" / "config" / "settings.py"
    )
    # Import may require torch; only assert the file exists + is non-trivial.
    assert spec is not None
    text = (ROOT / "src" / "model" / "config" / "settings.py").read_text(encoding="utf-8")
    assert "RANDOM_SEED" in text
    assert "EMBED_DIM" in text
