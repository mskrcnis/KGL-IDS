from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .config import DatasetConfig, output_root


RANDOM_STATE = 42
TEST_SIZE = 0.20
TIME_WINDOW_SECONDS = 5


def _print_distribution(series: pd.Series, title: str) -> None:
    print(f"\n{title}")
    print(series.value_counts(dropna=False))


def _safe_mode_fill(series: pd.Series, default: str = "unknown") -> pd.Series:
    modes = series.mode(dropna=True)
    return series.fillna(modes.iloc[0] if not modes.empty else default)


def _normalize_binary_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        values = set(pd.Series(series).dropna().unique().tolist())
        if values.issubset({0, 1}):
            return series.astype(int)
        return (series != 0).astype(int)

    values = series.astype(str).str.strip().str.lower()
    negative = {"0", "normal", "benign", "false", "no"}
    positive = {"1", "attack", "anomaly", "abnormal", "malicious", "intrusion", "true", "yes"}
    return values.map(lambda value: 0 if value in negative else 1 if value in positive else 1).astype(int)


def _clean_common(df: pd.DataFrame, cfg: DatasetConfig) -> pd.DataFrame:
    df = df.copy()

    required = [cfg.multiclass_target, cfg.binary_target, cfg.src_column, cfg.dst_column]
    if cfg.wustl:
        required.append("Dur")
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"{cfg.name}: missing required columns: {missing}")

    if cfg.wustl:
        zero_duration = int((df["Dur"] == 0).sum())
        print("Rows with Dur = 0:", zero_duration)
        df = df[df["Dur"] != 0].copy()

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    if not cfg.wustl:
        df.replace(["-", "--", "NA", "N/A", "na", "n/a", "", " "], np.nan, inplace=True)

    protected = {cfg.multiclass_target, cfg.binary_target}
    for column in df.columns:
        if column in protected or not pd.api.types.is_object_dtype(df[column]):
            continue
        converted = pd.to_numeric(df[column], errors="coerce")
        if converted.notna().mean() > 0.95:
            df[column] = converted

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns):
        df[numeric_columns] = df[numeric_columns].fillna(0)

    for column in df.select_dtypes(include=["object"]).columns:
        if column not in protected and df[column].isna().any():
            df[column] = _safe_mode_fill(df[column])

    for column in protected:
        if df[column].isna().any():
            default = 0 if column == cfg.binary_target else "unknown"
            df[column] = _safe_mode_fill(df[column], default=default)

    return df


def _edge_group(label: str) -> str:
    label = str(label).strip()
    if label in {"Normal", "BENIGN", "Benign", "normal"}:
        return "Normal"
    if label in {"DDoS_UDP", "DDoS_ICMP", "DDoS_TCP", "DDoS_HTTP"}:
        return "DDoS_DoS"
    if label in {"Vulnerability_scanner", "Port_Scanning", "Fingerprinting"}:
        return "Recon"
    if label in {"SQL_injection", "XSS", "Uploading"}:
        return "Web_Attack"
    if label == "Password":
        return "Brute_Force"
    if label == "Ransomware":
        return "Malware"
    if label in {"Backdoor", "MITM"}:
        return "Backdoor_MITM"
    return "Other"


def _add_targets(df: pd.DataFrame, cfg: DatasetConfig) -> tuple[pd.DataFrame, LabelEncoder]:
    df = df.copy()
    df["label_binary"] = _normalize_binary_target(df[cfg.binary_target])

    if cfg.wustl:
        df["attack_group"] = df[cfg.multiclass_target].astype(str).str.strip()
    else:
        df["attack_type_raw"] = df[cfg.multiclass_target].astype(str).str.strip()
        df["attack_group"] = df["attack_type_raw"].map(_edge_group)

    group_encoder = LabelEncoder()
    df["attack_group_enc"] = group_encoder.fit_transform(df["attack_group"].astype(str))
    _print_distribution(df["attack_group"], "Grouped attack distribution:")
    return df, group_encoder


def _encode_features(df: pd.DataFrame, cfg: DatasetConfig) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    df = df.copy()
    encoders: dict[str, LabelEncoder] = {}
    protected = {
        cfg.multiclass_target,
        cfg.binary_target,
        "attack_type_raw",
        "label_binary",
        "attack_group",
        "attack_group_enc",
    }

    if cfg.wustl:
        candidates = ["Proto"]
    else:
        candidates = [
            "http.request.method",
            "http.referer",
            "http.request.full_uri",
            "http.request.version",
            "http.response",
            "dns.qry.name",
            "mqtt.msg_decoded_as",
            "mqtt.msg",
            "mqtt.protoname",
            "mqtt.topic",
        ]
        candidates.extend(
            column for column in df.columns
            if column not in protected and not pd.api.types.is_numeric_dtype(df[column])
        )

    for column in sorted(set(candidates)):
        if column not in df.columns or column in protected:
            continue
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column].astype(str))
        encoders[column] = encoder

    return df, encoders


def _time_window(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series, errors="coerce")
    seconds = timestamps.astype("int64", copy=False) // 10**9
    seconds = pd.Series(seconds, index=series.index).where(~timestamps.isna(), np.nan)
    return (seconds // TIME_WINDOW_SECONDS).astype("Int64").fillna(-1).astype("int64")


def _group_count(df: pd.DataFrame, groups: list[str], name: str) -> None:
    df[name] = df.groupby(groups)[groups[0]].transform("size")


def _group_unique(df: pd.DataFrame, groups: list[str], target: str, name: str) -> None:
    df[name] = df.groupby(groups)[target].transform("nunique")


def _ratio(df: pd.DataFrame, numerator: str, denominator: str, name: str) -> None:
    denom = df[denominator].replace(0, np.nan)
    df[name] = (df[numerator] / denom).fillna(0.0)


def _build_wustl_kg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    event_time = pd.to_datetime(df["StartTime"], errors="coerce")
    event_time = event_time.fillna(pd.to_datetime(df["LastTime"], errors="coerce"))
    df["time_window"] = _time_window(event_time)

    src, dst = "SrcAddr", "DstAddr"
    _group_count(df, [src], "kg_src_flow_count")
    _group_count(df, [dst], "kg_dst_flow_count")
    _group_unique(df, [src], dst, "kg_src_unique_dsts")
    _group_unique(df, [dst], src, "kg_dst_unique_srcs")

    for column, src_name, dst_name in [
        ("Proto", "kg_src_unique_proto", "kg_dst_unique_proto"),
        ("Dport", "kg_src_unique_dports", "kg_dst_unique_dports"),
        ("Sport", "kg_src_unique_sports", "kg_dst_unique_sports"),
    ]:
        if column in df.columns:
            _group_unique(df, [src], column, src_name)
            _group_unique(df, [dst], column, dst_name)

    _group_count(df, [src, dst], "kg_src_dst_count")
    if "Proto" in df.columns:
        _group_count(df, [src, "Proto"], "kg_src_proto_count")
        _group_count(df, [dst, "Proto"], "kg_dst_proto_count")
        if "Dport" in df.columns:
            _group_count(df, ["Proto", "Dport"], "kg_proto_dport_count")
    if "Dport" in df.columns:
        _group_count(df, [src, "Dport"], "kg_src_dport_count")
        _group_count(df, [dst, "Dport"], "kg_dst_dport_count")
    if "Sport" in df.columns:
        _group_count(df, [src, "Sport"], "kg_src_sport_count")
        _group_count(df, [dst, "Sport"], "kg_dst_sport_count")

    _group_count(df, [src, "time_window"], "kg_src_window_count")
    _group_count(df, [dst, "time_window"], "kg_dst_window_count")
    if "Proto" in df.columns:
        _group_count(df, ["Proto", "time_window"], "kg_proto_window_count")
        _group_unique(df, [src, "time_window"], "Proto", "kg_src_unique_proto_window")
    if "Dport" in df.columns:
        _group_count(df, [dst, "Dport", "time_window"], "kg_dst_dport_window_count")
        _group_unique(df, [src, "time_window"], "Dport", "kg_src_unique_dports_window")
    _group_unique(df, [src, "time_window"], dst, "kg_src_unique_dsts_window")
    _group_unique(df, [dst, "time_window"], src, "kg_dst_unique_srcs_window")

    for source, prefix in [(src, "src"), (dst, "dst")]:
        if "Dur" in df.columns:
            df[f"kg_{prefix}_window_dur_sum"] = df.groupby([source, "time_window"])["Dur"].transform("sum")
        if "TotPkts" in df.columns:
            df[f"kg_{prefix}_window_pkt_sum"] = df.groupby([source, "time_window"])["TotPkts"].transform("sum")
        if "TotBytes" in df.columns:
            df[f"kg_{prefix}_window_byte_sum"] = df.groupby([source, "time_window"])["TotBytes"].transform("sum")

    _ratio(df, "kg_src_unique_dsts", "kg_src_flow_count", "kg_src_dst_diversity_ratio")
    _ratio(df, "kg_dst_unique_srcs", "kg_dst_flow_count", "kg_dst_src_diversity_ratio")
    _ratio(df, "kg_src_dst_count", "kg_src_flow_count", "kg_src_dst_focus_ratio")
    _ratio(df, "kg_src_window_count", "kg_src_flow_count", "kg_src_window_activity_ratio")
    _ratio(df, "kg_dst_window_count", "kg_dst_flow_count", "kg_dst_window_activity_ratio")
    if "kg_src_unique_dports" in df:
        _ratio(df, "kg_src_unique_dports", "kg_src_flow_count", "kg_src_dport_diversity_ratio")
    if "kg_src_unique_sports" in df:
        _ratio(df, "kg_src_unique_sports", "kg_src_flow_count", "kg_src_sport_diversity_ratio")
    if "kg_src_unique_proto" in df:
        _ratio(df, "kg_src_unique_proto", "kg_src_flow_count", "kg_src_proto_diversity_ratio")
    return df


def _build_edge_kg(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time_window"] = _time_window(df["frame.time"])
    src, dst = "ip.src_host", "ip.dst_host"

    proto_like = next(
        (column for column in ("http.request.method", "mqtt.msgtype", "dns.qry.type") if column in df.columns),
        None,
    )
    dport = next((column for column in ("tcp.dstport", "udp.port") if column in df.columns), None)

    _group_count(df, [src], "kg_src_flow_count")
    _group_count(df, [dst], "kg_dst_flow_count")
    _group_unique(df, [src], dst, "kg_src_unique_dsts")
    _group_unique(df, [dst], src, "kg_dst_unique_srcs")
    _group_count(df, [src, dst], "kg_src_dst_count")
    _group_count(df, [src, "time_window"], "kg_src_window_count")
    _group_count(df, [dst, "time_window"], "kg_dst_window_count")
    _group_unique(df, [src, "time_window"], dst, "kg_src_unique_dsts_window")

    if dport is not None:
        _group_unique(df, [src], dport, "kg_src_unique_dports")
        _group_unique(df, [dst], dport, "kg_dst_unique_dports")
        _group_count(df, [src, dport], "kg_src_dport_count")
        _group_count(df, [dst, dport], "kg_dst_dport_count")
        _group_count(df, [dst, dport, "time_window"], "kg_dst_dport_window_count")
        _group_unique(df, [src, "time_window"], dport, "kg_src_unique_dports_window")

    if proto_like is not None:
        _group_unique(df, [src], proto_like, "kg_src_unique_proto_like")
        _group_count(df, [src, proto_like], "kg_src_proto_like_count")
        _group_count(df, [proto_like, "time_window"], "kg_proto_like_window_count")

    _ratio(df, "kg_src_unique_dsts", "kg_src_flow_count", "kg_src_dst_diversity_ratio")
    _ratio(df, "kg_dst_unique_srcs", "kg_dst_flow_count", "kg_dst_src_diversity_ratio")
    _ratio(df, "kg_src_dst_count", "kg_src_flow_count", "kg_src_dst_focus_ratio")
    _ratio(df, "kg_src_window_count", "kg_src_flow_count", "kg_src_window_activity_ratio")
    if "kg_src_unique_dports" in df:
        _ratio(df, "kg_src_unique_dports", "kg_src_flow_count", "kg_src_dport_diversity_ratio")
    return df


def _feature_columns(df: pd.DataFrame, cfg: DatasetConfig) -> list[str]:
    excluded = {
        cfg.multiclass_target,
        cfg.binary_target,
        "attack_type_raw",
        "label_binary",
        "attack_group",
        "attack_group_enc",
    }
    columns = [column for column in df.columns if column not in excluded]
    non_numeric = [column for column in columns if not pd.api.types.is_numeric_dtype(df[column])]
    if non_numeric:
        raise ValueError(f"Non-numeric features remain: {non_numeric}")
    return columns


def _save_split(df: pd.DataFrame, cfg: DatasetConfig, out_dir: Path) -> None:
    features = _feature_columns(df, cfg)
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["attack_group"],
    )

    scaler = StandardScaler()
    train_x = pd.DataFrame(scaler.fit_transform(train_df[features]), columns=features)
    test_x = pd.DataFrame(scaler.transform(test_df[features]), columns=features)

    train_group = train_x.copy()
    train_group["attack_group_enc"] = train_df["attack_group_enc"].to_numpy()
    test_group = test_x.copy()
    test_group["attack_group_enc"] = test_df["attack_group_enc"].to_numpy()

    out_dir.mkdir(parents=True, exist_ok=True)
    train_group.to_parquet(out_dir / "train_group.parquet", index=False)
    test_group.to_parquet(out_dir / "test_group.parquet", index=False)
    train_df.to_parquet(out_dir / "train_unscaled.parquet", index=False)
    test_df.to_parquet(out_dir / "test_unscaled.parquet", index=False)
    joblib.dump(scaler, out_dir / "scaler.pkl")
    joblib.dump(features, out_dir / "feature_columns.pkl")


def _prepare_variant(raw: pd.DataFrame, cfg: DatasetConfig, variant: str, root: Path) -> None:
    df = _clean_common(raw, cfg)
    if variant == "baseline":
        df.drop(columns=[c for c in cfg.baseline_drop_columns if c in df.columns], inplace=True)
    elif variant == "kg":
        df = _build_wustl_kg(df) if cfg.wustl else _build_edge_kg(df)
        df.drop(columns=[c for c in cfg.kg_late_drop_columns if c in df.columns], inplace=True)
    else:
        raise ValueError(f"Unsupported preprocessing variant: {variant}")

    df, encoders = _encode_features(df, cfg)
    df, group_encoder = _add_targets(df, cfg)

    out_dir = root / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "cleaned.parquet", index=False)
    joblib.dump(encoders, out_dir / "feature_encoders.pkl")
    joblib.dump(group_encoder, out_dir / "attack_group_encoder.pkl")
    pd.DataFrame({
        "attack_group": group_encoder.classes_,
        "attack_group_enc": range(len(group_encoder.classes_)),
    }).to_csv(out_dir / "attack_group_mapping.csv", index=False)
    _save_split(df, cfg, out_dir)


def run_preprocessing(cfg: DatasetConfig) -> None:
    if not cfg.raw_path.exists():
        raise FileNotFoundError(f"Raw dataset not found: {cfg.raw_path}")
    raw = pd.read_csv(cfg.raw_path, low_memory=False)
    root = output_root(cfg)
    print(f"Dataset: {cfg.name}")
    print(f"Raw path: {cfg.raw_path}")
    print(f"Raw shape: {raw.shape}")
    _prepare_variant(raw, cfg, "baseline", root)
    _prepare_variant(raw, cfg, "kg", root)
    print(f"Preprocessing outputs written to: {root}")
