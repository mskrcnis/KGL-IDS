from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from kgl_ids.config import get_dataset, output_root  # noqa: E402
from kgl_ids.model import train_and_evaluate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the article ANN")
    parser.add_argument("--dataset", choices=["wustl", "edgeiiot"], required=True)
    parser.add_argument(
        "--variant",
        choices=["baseline", "kg", "kg_smote", "kgl_ids"],
        default="kgl_ids",
        help="Ablation variant or full KGL-IDS pipeline",
    )
    args = parser.parse_args()

    root = output_root(get_dataset(args.dataset))
    encoder_path = root / "kg" / "attack_group_encoder.pkl"
    if args.variant == "baseline":
        data_dir = root / "baseline"
        train_path, test_path = data_dir / "train_group.parquet", data_dir / "test_group.parquet"
    elif args.variant == "kg":
        data_dir = root / "kg"
        train_path, test_path = data_dir / "train_group.parquet", data_dir / "test_group.parquet"
    elif args.variant == "kg_smote":
        data_dir = root / "smote"
        train_path, test_path = data_dir / "train_group_smote.parquet", data_dir / "test_group.parquet"
    else:
        data_dir = root / "selected"
        train_path = data_dir / "train_df_group_selected.parquet"
        test_path = data_dir / "test_df_group_selected.parquet"

    train_and_evaluate(
        train_path=train_path,
        test_path=test_path,
        encoder_path=encoder_path,
        output_dir=root / "models" / args.variant,
    )


if __name__ == "__main__":
    main()
