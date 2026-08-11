from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kgl_ids.balancing import run_smote  # noqa: E402
from kgl_ids.config import get_dataset, output_root  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply SMOTE to the KG training split only")
    parser.add_argument("--dataset", choices=["wustl", "edgeiiot"], required=True)
    args = parser.parse_args()
    root = output_root(get_dataset(args.dataset))
    run_smote(root / "kg", root / "smote")


if __name__ == "__main__":
    main()
