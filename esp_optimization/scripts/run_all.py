from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import run_all


if __name__ == "__main__":
    result = run_all(PROJECT_ROOT / "config" / "model.yaml")
    print(json.dumps(result, ensure_ascii=False, indent=2))

