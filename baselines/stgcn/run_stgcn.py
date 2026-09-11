"""Entry point for stgcn; works without a custom PYTHONPATH."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.stgcn.train import main

if __name__ == "__main__":
    main()
