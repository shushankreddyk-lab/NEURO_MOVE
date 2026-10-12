from __future__ import annotations

import os
import sys
from pathlib import Path

from neuromove.config import PROJECT_ROOT, SRC_DIR


def ensure_engine_imports() -> None:
    """Make the existing BCI engines importable from the release backend."""
    src_abs = str(SRC_DIR.resolve())
    if src_abs not in sys.path:
        sys.path.insert(0, src_abs)

    project_abs = str(PROJECT_ROOT.resolve())
    if project_abs not in sys.path:
        sys.path.insert(0, project_abs)


def import_engines() -> dict[str, object]:
    """Import the engines we intend to expose via the API.

    Returns a mapping from engine name to the imported module.
    """
    ensure_engine_imports()

    engines: dict[str, object] = {}
    try:
        import minirocket_engine as mre  # type: ignore[import-not-found]

        engines["minirocket"] = mre
    except Exception as exc:  # noqa: BLE001
        engines["minirocket"] = {"error": repr(exc)}

    try:
        import advanced_eeg_engine as aee  # type: ignore[import-not-found]

        engines["conformer"] = aee
    except Exception as exc:  # noqa: BLE001
        engines["conformer"] = {"error": repr(exc)}

    try:
        import cnn_lstm_engine as cle  # type: ignore[import-not-found]

        engines["cnn_lstm"] = cle
    except Exception as exc:  # noqa: BLE001
        engines["cnn_lstm"] = {"error": repr(exc)}

    try:
        import eegnet_engine as ene  # type: ignore[import-not-found]

        engines["eegnet"] = ene
    except Exception as exc:  # noqa: BLE001
        engines["eegnet"] = {"error": repr(exc)}

    try:
        import convnets_engine as coe  # type: ignore[import-not-found]

        engines["convnets"] = coe
    except Exception as exc:  # noqa: BLE001
        engines["convnets"] = {"error": repr(exc)}

    return engines
