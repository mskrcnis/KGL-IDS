from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tensorflow import keras
from tensorflow.keras import layers


RANDOM_STATE = 123
EPOCHS = 50
BATCH_SIZE = 4096
LEARNING_RATE = 1e-3
DROPOUT = 0.3
PATIENCE = 7
VALIDATION_SPLIT = 0.1


def _set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_ann(input_dim: int, class_count: int) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(256),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(DROPOUT),
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(DROPOUT),
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Dropout(DROPOUT),
        layers.Dense(class_count, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_and_evaluate(
    train_path: Path,
    test_path: Path,
    encoder_path: Path,
    output_dir: Path,
) -> None:
    _set_seed()
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    target = "attack_group_enc"
    feature_names = [column for column in train_df.columns if column != target]

    x_train = train_df[feature_names].to_numpy(dtype=np.float32)
    y_train = train_df[target].to_numpy(dtype=np.int32)
    x_test = test_df[feature_names].to_numpy(dtype=np.float32)
    y_test = test_df[target].to_numpy(dtype=np.int32)
    encoder = joblib.load(encoder_path)
    class_names = list(encoder.classes_)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "best_ann.keras"
    model = build_ann(x_train.shape[1], len(class_names))
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            str(model_path), monitor="val_loss", save_best_only=True, verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
        ),
    ]

    history = model.fit(
        x_train,
        y_train,
        validation_split=VALIDATION_SPLIT,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )
    pd.DataFrame(history.history).assign(epoch=np.arange(1, len(history.history["loss"]) + 1)).to_csv(
        output_dir / "training_history.csv", index=False
    )

    model = keras.models.load_model(model_path)
    test_loss, test_accuracy = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
    probabilities = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
        y_test, predictions, average="weighted", zero_division=0
    )
    precision_m, recall_m, f1_m, _ = precision_recall_fscore_support(
        y_test, predictions, average="macro", zero_division=0
    )
    report_text = classification_report(
        y_test, predictions, target_names=class_names, zero_division=0
    )
    report_dict = classification_report(
        y_test, predictions, target_names=class_names, output_dict=True, zero_division=0
    )
    matrix = confusion_matrix(y_test, predictions)

    (output_dir / "classification_report.txt").write_text(report_text, encoding="utf-8")
    pd.DataFrame(report_dict).transpose().to_csv(output_dir / "classification_report.csv")
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(output_dir / "confusion_matrix.csv")
    pd.DataFrame({"y_true": y_test, "y_pred": predictions}).to_csv(
        output_dir / "test_predictions.csv", index=False
    )

    metrics = {
        "train_path": str(train_path),
        "test_path": str(test_path),
        "num_features": len(feature_names),
        "class_names": class_names,
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "precision_weighted": float(precision_w),
        "recall_weighted": float(recall_w),
        "f1_weighted": float(f1_w),
        "precision_macro": float(precision_m),
        "recall_macro": float(recall_m),
        "f1_macro": float(f1_m),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
