from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    raw_path: Path
    multiclass_target: str
    binary_target: str
    raw_attack_target: str | None
    time_columns: tuple[str, ...]
    src_column: str
    dst_column: str
    baseline_drop_columns: tuple[str, ...]
    kg_late_drop_columns: tuple[str, ...]
    wustl: bool = False


DATASETS = {
    "wustl": DatasetConfig(
        name="wustl",
        raw_path=DATA_ROOT / "raw" / "wustl" / "wustl_corrected.csv",
        multiclass_target="Traffic",
        binary_target="Target",
        raw_attack_target=None,
        time_columns=("StartTime", "LastTime"),
        src_column="SrcAddr",
        dst_column="DstAddr",
        baseline_drop_columns=("StartTime", "LastTime", "SrcAddr", "DstAddr"),
        kg_late_drop_columns=("StartTime", "LastTime", "SrcAddr", "DstAddr"),
        wustl=True,
    ),
    "edgeiiot": DatasetConfig(
        name="edgeiiot",
        raw_path=DATA_ROOT / "raw" / "edgeiiot" / "DNN-EdgeIIoT-dataset.csv",
        multiclass_target="Attack_type",
        binary_target="Attack_label",
        raw_attack_target="Attack_type",
        time_columns=("frame.time",),
        src_column="ip.src_host",
        dst_column="ip.dst_host",
        baseline_drop_columns=(
            "frame.time",
            "ip.src_host",
            "ip.dst_host",
            "arp.src.proto_ipv4",
            "arp.dst.proto_ipv4",
        ),
        kg_late_drop_columns=(
            "frame.time",
            "ip.src_host",
            "ip.dst_host",
            "arp.src.proto_ipv4",
            "arp.dst.proto_ipv4",
        ),
        wustl=False,
    ),
}


def get_dataset(name: str) -> DatasetConfig:
    try:
        return DATASETS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(DATASETS))
        raise ValueError(f"Unknown dataset '{name}'. Choose one of: {valid}") from exc


def output_root(dataset: DatasetConfig) -> Path:
    return DATA_ROOT / "processed" / dataset.name
