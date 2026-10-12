from __future__ import annotations

import os
from pathlib import Path

# Root of the release repo
REPO_ROOT = Path(__file__).resolve().parents[1]

# The existing project tree that contains the BCI engines and datasets.
# This is the real runtime source of truth for the current project.
PROJECT_ROOT = Path(r"D:\eeg-minirocket-project")

# Where the existing engines live.
SRC_DIR = PROJECT_ROOT / "eeg-mi-bci-archive-565657a" / "src"

# Existing virtual environment with torch/mne/sktime/streamlit etc.
VENV_PYTHON = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"

# Shared storage for backend artifacts
ARTIFACT_ROOT = REPO_ROOT / "artifacts"
UPLOAD_ROOT = REPO_ROOT / "uploads"


def ensure_paths() -> None:
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


def is_venv_python_reachable() -> bool:
    return VENV_PYTHON.exists()
