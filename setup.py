"""Packaging for INS-HDGS-CMT.

Installs the source tree under src/ (the `model` and `data_pipeline` packages)
in editable mode so scripts and tests can import them:

    pip install -e .

PyTorch is intentionally NOT listed as an install dependency here — install it
separately, matching your machine's CUDA version (see README / requirements.txt).
"""
from pathlib import Path

from setuptools import find_packages, setup

ROOT = Path(__file__).parent
long_description = (ROOT / "README.md").read_text(encoding="utf-8")


def _requirements():
    reqs = []
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            reqs.append(line)
    return reqs


setup(
    name="ins-hdgs-cmt",
    version="1.0.0",
    description=(
        "Interpretable Neuro-Symbolic Hybrid Dynamic Graph Spiking Cross-Modal "
        "Transformer for consumer engagement prediction from EEG and eye tracking."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Priyadharshini Dhanapalan and the INS-HDGS-CMT authors",
    url="https://github.com/priyadharshini-D29/INS-HDGS-CMT",
    license="MIT",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11",
    install_requires=_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
)
