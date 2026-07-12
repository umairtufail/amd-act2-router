"""Compatibility entrypoint for ``streamlit run demo_app.py``."""

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo.app import render_app

render_app()
