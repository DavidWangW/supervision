"""Supervision video analysis API."""

import sys
from pathlib import Path

# Ensure the *local* project ``supervision`` (repo root ``src/supervision``) is
# used in preference to any PyPI-installed copy. The service depends on features
# and modifications that live in this repository's ``src/supervision``, so when
# the app is launched with an interpreter that does not have the editable
# install active (e.g. a system Python or an IDE-managed venv), we prepend the
# repo's ``src`` directory to ``sys.path``. This must run before any module in
# this package does ``import supervision``.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LOCAL_SV_SRC = str(_REPO_ROOT / "src")
if _LOCAL_SV_SRC not in sys.path:
    sys.path.insert(0, _LOCAL_SV_SRC)
