from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from imblearn.over_sampling import SMOTE


RANDOM_STATE = 42
K_NEIGHBORS = 5


def run_smote(kg_dir: Path, output_dir: Path) -> None:
    train_path = kg_dir / "train_group.parquet"
    test_path = kg_dir / "test_group.parquet"
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    target = "attack_group_enc"

    if target not in train_df.columns or target not in test_df.columns:
        raise ValueError(f"Expected '{target}' in both train and test data")

    x_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=K_NEIGHBORS)
    x_balanced, y_balanced = smote.fit_resample(x_train, y_train)

    balanced = pd.DataFrame(x_balanced, columns=x_train.columns)
    balanced[target] = y_balanced
    output_dir.mkdir(parents=True, exist_ok=True)
    balanced.to_parquet(output_dir / "train_group_smote.parquet", index=False)
    test_df.to_parquet(output_dir / "test_group.parquet", index=False)

    summary = {
        "method": "SMOTE on training data only",
        "random_state": RANDOM_STATE,
        "k_neighbors": K_NEIGHBORS,
        "input_train": str(train_path),
        "input_test": str(test_path),
        "train_shape_before": list(train_df.shape),
        "train_shape_after": list(balanced.shape),
        "test_shape": list(test_df.shape),
        "class_distribution_before": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
        "class_distribution_after": {str(k): int(v) for k, v in balanced[target].value_counts().sort_index().items()},
    }
    with (output_dir / "smote_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
