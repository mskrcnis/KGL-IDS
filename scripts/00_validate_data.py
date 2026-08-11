from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kgl_ids.config import get_dataset  # noqa: E402


EXPECTED_COLUMNS = {
    "wustl": {"StartTime", "LastTime", "SrcAddr", "DstAddr", "Proto", "Dur", "Traffic", "Target"},
    "edgeiiot": {"frame.time", "ip.src_host", "ip.dst_host", "Attack_type", "Attack_label"},
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate article dataset placement and schema")
    parser.add_argument("--dataset", choices=["wustl", "edgeiiot"], required=True)
    args = parser.parse_args()
    config = get_dataset(args.dataset)

    print(f"Expected file: {config.raw_path}")
    if not config.raw_path.exists():
        raise FileNotFoundError(
            f"Dataset is missing. Read data/README.md and place the file at: {config.raw_path}"
        )

    with config.raw_path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = set(next(csv.reader(handle)))
    missing = sorted(EXPECTED_COLUMNS[args.dataset] - header)
    if missing:
        raise ValueError(
            f"{args.dataset} input is missing required columns: {missing}. "
            "For WUSTL, use the corrected CSV required by the article pipeline."
        )
    print(f"Validated {args.dataset}: {len(header)} columns found.")


if __name__ == "__main__":
    main()
