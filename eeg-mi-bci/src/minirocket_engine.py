"""
MiniRocketPipeline - Unified implementation from archive.
This is a thin wrapper importing from the canonical archive implementation.
"""
import sys
from pathlib import Path

# Ensure archive src is in path
_archive_src = Path(__file__).resolve().parents[2] / "eeg-mi-bci-archive-565657a" / "src"
if str(_archive_src) not in sys.path:
    sys.path.insert(0, str(_archive_src))

from minirocket_engine import MiniRocketPipeline as _ArchivePipeline
from minirocket_engine import TorchMiniRocket, GPUMiniRocketHead, get_device

# Re-export for backward compatibility
__all__ = ['MiniRocketPipeline', 'TorchMiniRocket', 'GPUMiniRocketHead', 'get_device']

MiniRocketPipeline = _ArchivePipeline