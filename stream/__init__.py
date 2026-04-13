from __future__ import annotations

import os
from pathlib import Path

__version__ = "0.0.8"


def _ensure_matplotlib_cache_dir() -> None:
    cache_dir = Path("/tmp/matplotlib")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


_ensure_matplotlib_cache_dir()
