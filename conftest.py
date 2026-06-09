"""Pytest configuration — add repo root to sys.path for module resolution."""
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `data`, `training`, etc. are importable
sys.path.insert(0, str(Path(__file__).parent))
