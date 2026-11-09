"""Load the numbered V3 configuration through an importable bridge."""
from __future__ import annotations

import neuropsyche_v3_config as cfg

if hasattr(cfg, "apply_persisted_settings"):
    cfg.apply_persisted_settings()
