from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MinMaxScaler


RANDOM_STATE = 42
SAMPLE_N = 10_000_000
POPULATION_SIZE = 30
NUM_ITERATIONS = 50
GWO_A_START = 2.0
GWO_A_END = 0.0
W_MAX = 0.9
W_MIN = 0.4
C1 = 1.8
C2 = 1.8
BIN_THRESHOLD = 0.5
PENALTY_LAMBDA = 0.995


def compute_fisher_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    classes = np.unique(y)
    overall_mean = x.mean(axis=0)
    numerator = np.zeros(x.shape[1], dtype=np.float64)
    denominator = np.zeros(x.shape[1], dtype=np.float64)

    for class_id in classes:
        mask = y == class_id
        count = int(mask.sum())
        class_x = x[mask]
        class_mean = class_x.mean(axis=0)
        class_var = class_x.var(axis=0)
        numerator += count * (class_mean - overall_mean) ** 2
        denominator += count * class_var

    scores = np.where(denominator > 0, numerator / denominator, 0.0)
    minimum, maximum = scores.min(), scores.max()
    if maximum > minimum:
        scores = (scores - minimum) / (maximum - minimum)
    return scores.astype(np.float64)


def _binary_from_position(position: np.ndarray) -> np.ndarray:
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(position, -20, 20)))
    bits = (probabilities >= BIN_THRESHOLD).astype(np.int32)
    if bits.sum() == 0:
        bits[np.argmax(probabilities)] = 1
    return bits


def _fitness(bits: np.ndarray, fisher: np.ndarray, correlation: np.ndarray) -> float:
    selected = np.where(bits == 1)[0]
    if len(selected) == 0:
        return -1e18

    fisher_sum = float(fisher[selected].sum())
    correlation_penalty = 0.0
    for i, left in enumerate(selected):
        for right in selected[i + 1:]:
            correlation_penalty += float(correlation[left, right])
    return PENALTY_LAMBDA * fisher_sum - (1.0 - PENALTY_LAMBDA) * correlation_penalty


def _select(fisher: np.ndarray, correlation: np.ndarray) -> tuple[list[int], float]:
    rng = np.random.default_rng(RANDOM_STATE)
    feature_count = len(fisher)
    positions = rng.uniform(-2.0, 2.0, (POPULATION_SIZE, feature_count))
    velocities = rng.uniform(-0.5, 0.5, (POPULATION_SIZE, feature_count))
    personal_positions = positions.copy()
    personal_scores = np.full(POPULATION_SIZE, -1e18)
    cache: dict[tuple[int, ...], float] = {}

    def evaluate(position: np.ndarray) -> float:
        bits = _binary_from_position(position)
        key = tuple(int(bit) for bit in bits)
        if key not in cache:
            cache[key] = _fitness(bits, fisher, correlation)
        return cache[key]

    for index in range(POPULATION_SIZE):
        personal_scores[index] = evaluate(positions[index])

    def leaders() -> tuple[int, int, int]:
        order = np.argsort(-personal_scores)
        return int(order[0]), int(order[1]), int(order[2])

    alpha, beta, delta = leaders()
    best_position = personal_positions[alpha].copy()
    best_score = float(personal_scores[alpha])

    for iteration in range(NUM_ITERATIONS):
        progress = iteration / max(1, NUM_ITERATIONS - 1)
        a = GWO_A_START - (GWO_A_START - GWO_A_END) * progress
        inertia = W_MAX - (W_MAX - W_MIN) * progress

        for index in range(POPULATION_SIZE):
            current = positions[index]
            guides = []
            for leader_index in (alpha, beta, delta):
                r1 = rng.random(feature_count)
                r2 = rng.random(feature_count)
                a_vec = 2 * a * r1 - a
                c_vec = 2 * r2
                leader_position = personal_positions[leader_index]
                distance = np.abs(c_vec * leader_position - current)
                guides.append(leader_position - a_vec * distance)
            gwo_position = sum(guides) / 3.0

            rp = rng.random(feature_count)
            rg = rng.random(feature_count)
            velocities[index] = (
                inertia * velocities[index]
                + C1 * rp * (personal_positions[index] - current)
                + C2 * rg * (gwo_position - current)
            )
            positions[index] = np.clip(current + velocities[index], -6.0, 6.0)
            score = evaluate(positions[index])
            if score > personal_scores[index]:
                personal_scores[index] = score
                personal_positions[index] = positions[index].copy()

        alpha, beta, delta = leaders()
        if personal_scores[alpha] > best_score:
            best_score = float(personal_scores[alpha])
            best_position = personal_positions[alpha].copy()

    selected = np.where(_binary_from_position(best_position) == 1)[0].tolist()
    return selected, best_score


def run_selection(smote_dir: Path, output_dir: Path) -> None:
    train_df = pd.read_parquet(smote_dir / "train_group_smote.parquet")
    test_df = pd.read_parquet(smote_dir / "test_group.parquet")
    label_col = "attack_group_enc"
    feature_names = [column for column in train_df.columns if column != label_col]

    x_train = train_df[feature_names].to_numpy(dtype=np.float32)
    y_train = train_df[label_col].to_numpy(dtype=np.int32)
    x_test = test_df[feature_names].to_numpy(dtype=np.float32)

    if SAMPLE_N < len(y_train):
        splitter = StratifiedShuffleSplit(n_splits=1, train_size=SAMPLE_N, random_state=RANDOM_STATE)
        sample_indices, _ = next(splitter.split(x_train, y_train))
        x_sample, y_sample = x_train[sample_indices], y_train[sample_indices]
    else:
        x_sample, y_sample = x_train, y_train

    scaled_sample = MinMaxScaler().fit_transform(x_sample)
    fisher = compute_fisher_scores(scaled_sample, y_sample)
    correlation = np.abs(np.corrcoef(scaled_sample, rowvar=False))
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 0.0)

    selected_indices, best_score = _select(fisher, correlation)
    selected_names = [feature_names[index] for index in selected_indices]

    output_dir.mkdir(parents=True, exist_ok=True)
    train_df[selected_names + [label_col]].to_parquet(
        output_dir / "train_df_group_selected.parquet", index=False
    )
    test_df[selected_names + [label_col]].to_parquet(
        output_dir / "test_df_group_selected.parquet", index=False
    )
    np.save(output_dir / "correlation_matrix.npy", correlation)
    pd.DataFrame({"feature": feature_names, "fisher_score": fisher}).sort_values(
        "fisher_score", ascending=False
    ).to_csv(output_dir / "fisher_scores.csv", index=False)

    metadata = {
        "method": "Hybrid-GWO-PSO-FisherCorr-Filter",
        "input_train": str(smote_dir / "train_group_smote.parquet"),
        "input_test": str(smote_dir / "test_group.parquet"),
        "sample_n": int(min(SAMPLE_N, len(y_train))),
        "num_features_total": len(feature_names),
        "selected_feature_count": len(selected_names),
        "selected_feature_indices": selected_indices,
        "selected_feature_names": selected_names,
        "best_fitness": best_score,
        "params": {
            "num_iterations": NUM_ITERATIONS,
            "population_size": POPULATION_SIZE,
            "w_max": W_MAX,
            "w_min": W_MIN,
            "c1": C1,
            "c2": C2,
            "bin_threshold": BIN_THRESHOLD,
            "penalty_lambda": PENALTY_LAMBDA,
            "random_state": RANDOM_STATE,
        },
        "fitness_definition": "penalty_lambda * fisher_sum - (1 - penalty_lambda) * correlation_penalty",
    }
    with (output_dir / "selected_features.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
