"""Boilerplate shared by every numbered entry point in this directory.

Importing this module makes ``contamination_audit`` importable when running
scripts directly (``python scripts/01_ngram_filter.py``) without installing
the package. Also exposes the repo-root path so scripts can resolve configs.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Make ``contamination_audit`` importable when running scripts ad-hoc.
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Convenience: canonical default config path for every script.
DEFAULT_CONFIG = REPO_ROOT / "configs" / "thresholds.yaml"
DATASETS_CONFIG = REPO_ROOT / "configs" / "datasets.yaml"
MODELS_CONFIG = REPO_ROOT / "configs" / "models.yaml"
