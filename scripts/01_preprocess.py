from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kgl_ids.config import get_dataset  # noqa: E402
from kgl_ids.preprocessing import run_preprocessing  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run article-faithful baseline and KG preprocessing")
    parser.add_argument("--dataset", choices=["wustl", "edgeiiot"], required=True)
    args = parser.parse_args()
    run_preprocessing(get_dataset(args.dataset))


if __name__ == "__main__":
    main()
