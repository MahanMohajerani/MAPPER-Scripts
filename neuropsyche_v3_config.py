"""Import the numbered configuration file for worker processes."""
from __future__ import annotations

from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent / "0_project_config_V3.py"
if not _CONFIG_PATH.exists():
    raise ImportError(f"Could not find configuration script: {_CONFIG_PATH}")

_code = compile(_CONFIG_PATH.read_bytes(), str(_CONFIG_PATH), "exec")
exec(_code, globals(), globals())

if "apply_persisted_settings" in globals():
    apply_persisted_settings()
