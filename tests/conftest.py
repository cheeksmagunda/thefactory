"""Shared pytest config."""

import os
import sys
from pathlib import Path

# Make `src` importable without installing.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")
os.environ.setdefault("DISABLE_SCHEDULER", "true")
