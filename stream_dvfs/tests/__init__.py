"""Utility scripts for validating legacy DVFS and parsing workflows."""

from __future__ import annotations

import os
from pathlib import Path

_matplotlib_dir = Path("/tmp/matplotlib")
_matplotlib_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_dir))
