from __future__ import annotations

import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
WORKSPACE_ROOT = REPO_ROOT.parent

CONFIG_DIR = PACKAGE_ROOT / "config"
CORE_CONFIG_DIR = CONFIG_DIR / "cores"
DVFS_CONFIG_DIR = CONFIG_DIR / "dvfs"
MAPPING_CONFIG_DIR = CONFIG_DIR / "mappings"
MULTICORE_CONFIG_DIR = CONFIG_DIR / "multicores"

DOCS_DIR = PACKAGE_ROOT / "docs"
EXPERIMENTS_DIR = PACKAGE_ROOT / "experiments"
OUTPUTS_DIR = PACKAGE_ROOT / "outputs"


def ensure_gurobi_license() -> Path:
    license_path = WORKSPACE_ROOT / "gurobi.lic"
    if license_path.exists():
        os.environ.setdefault("GRB_LICENSE_FILE", str(license_path))
    return license_path


def ensure_output_dir(*parts: str) -> Path:
    output_dir = OUTPUTS_DIR.joinpath(*parts)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
