"""
evaluation - Unified implementation from archive.
This is a thin wrapper importing from the canonical archive implementation.
"""
import sys
from pathlib import Path

# Ensure archive src is in path
_archive_src = Path(__file__).resolve().parents[2] / "eeg-mi-bci-archive-565657a" / "src"
if str(_archive_src) not in sys.path:
    sys.path.insert(0, str(_archive_src))

from evaluation import *

# Re-export for backward compatibility
__all__ = [name for name in dir() if not name.startswith('_')]
