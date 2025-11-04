"""Initialization helpers for the factor workflow example package."""

from __future__ import annotations

import sys
from pathlib import Path


_PACKAGE_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PACKAGE_ROOT.parent.parent
_path_str = str(_REPO_ROOT)
if _path_str not in sys.path:
	sys.path.insert(0, _path_str)

__all__ = []
