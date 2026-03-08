from pathlib import Path
import sys


NOTEBOOKS_DIR = Path(__file__).resolve().parent / "Notebooks"
if NOTEBOOKS_DIR.exists():
    sys.path.insert(0, str(NOTEBOOKS_DIR))
