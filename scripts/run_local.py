"""
Run AISmartMirror locally (development laptop or Pi).

Ensures project root is in path and invokes app.main.
Usage: python scripts/run_local.py
       or: python -m app.main (from project root)
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.main import main

if __name__ == "__main__":
    main()
