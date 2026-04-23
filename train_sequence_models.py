from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
from collections import Counter
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit

from meditation_pipeline import FEATURE_NAMES_19, MeditationFeaturePipeline, align_features


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Experimental LSTM/Bi-LSTM training on window sequences")
    p.add_argument("--input", nargs="+", required=True, help="Raw CSV files")
    p.add_argument("--label-col", default="label")
    p.add_argument("--subject-col", default="subject_id")
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--step", type=int, default=5)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out-dir", default="models")
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def window_labels_from_raw(df: pd.DataFrame, meta_df: pd.DataFrame, label_col: str) -> np.ndarray:
    labels = []
    for _, m in meta_df.iterrows():
        vals = pd.to_numeric(df[label_col].iloc[int(m.start_idx):int(m.end_idx)], errors="coerce").dropna()
        if len(vals) == 0:
            labels.append(np.nan)
            continue
        labels.append(vals.mode().iloc[0])
    return np.asarray(labels)


def create_sequences(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seq_len: int,
    step: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, gs = [], [], []
    uniq = np.unique(groups)
    for g in uniq:
        idx = np.where(groups == g)[0]
        xg, yg = x[idx], y[idx]
        for i in range(0, len(xg) - seq_len + 1, step):
            seq_x = xg[i : i + seq_len]
            most_common_class = Counter(yg[i : i + seq_len]).most_common(1)[0][0]
            xs.append(seq_x)
            ys.append(most_common_class)
            gs.append(g)
    return np.asarray(xs), np.asarray(ys).astype(int), np.asarray(gs)


def build_dataset(files: List[str], label_col: str, subject_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pipe = MeditationFeaturePipeline()
    rows, labels, groups = [], [], []

    for path in files:
        df = pd.read_csv(path, low_memory=False)
        if label_col not in df.columns:
            continue

        feat_df, meta_df, _ = pipe.extract(df, drop_gated_windows=True)
        if feat_df.empty:
            continue

        y = window_labels_from_raw(df, meta_df, label_col)
        keep = np.isfinite(y)
        if keep.sum() == 0:
            continue

        feat_df = feat_df.loc[keep].reset_index(drop=True)
        y = y[keep].astype(int)

        subj = str(df[subject_col].dropna().iloc[0]) if subject_col in df.columns and df[subject_col].notna().any() else os.path.splitext(os.path.basename(path))[0]

        rows.append(align_features(feat_df, FEATURE_NAMES_19).values)
        labels.append(y)
        groups.append(np.array([subj] * len(y)))

    if not rows:
        raise RuntimeError("No sequence-ready data produced")

    X = np.concatenate(rows)
    y = np.concatenate(labels)
    g = np.concatenate(groups)
    return X, y, g


def build_lstm(input_shape, n_classes: int, bidirectional: bool):
    import tensorflow as tf
    from tensorflow.keras import Sequential
    from tensorflow.keras.layers import Bidirectional, Dense, Dropout, LSTM

    model = Sequential()
    if bidirectional:
        model.add(Bidirectional(LSTM(64, return_sequences=False), input_shape=input_shape))
    else:
        model.add(LSTM(64, return_sequences=False, input_shape=input_shape))
    model.add(Dropout(0.4))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(n_classes, activation="softmax"))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if importlib.util.find_spec("tensorflow") is None:  # pragma: no cover
        raise RuntimeError("TensorFlow is required for LSTM/Bi-LSTM experiments")
    import tensorflow as tf

    X_win, y_win, g_win = build_dataset(args.input, args.label_col, args.subject_col)
    X_seq, y_seq, g_seq = create_sequences(X_win, y_win, g_win, seq_len=args.seq_len, step=args.step)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=args.random_state)
    tr, va = next(splitter.split(X_seq, y_seq, groups=g_seq))

    X_train, X_val = X_seq[tr], X_seq[va]
    y_train, y_val = y_seq[tr], y_seq[va]

    n_classes = int(np.max(y_seq)) + 1
    input_shape = (X_train.shape[1], X_train.shape[2])

    classes, counts = np.unique(y_train, return_counts=True)
    class_weight = {int(c): float(len(y_train) / (len(classes) * cnt)) for c, cnt in zip(classes, counts)}

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True)
    ]

    results = {}
    for name, bidirectional in [("lstm", False), ("bilstm", True)]:
        model = build_lstm(input_shape, n_classes=n_classes, bidirectional=bidirectional)
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            class_weight=class_weight,
            callbacks=callbacks,
            verbose=1,
        )

        proba = model.predict(X_val, verbose=0)
        pred = np.argmax(proba, axis=1)

        results[name] = {
            "macro_f1": float(f1_score(y_val, pred, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y_val, pred)),
        }

        model.save(os.path.join(args.out_dir, f"{name}_sequence_model.keras"))

    with open(os.path.join(args.out_dir, "sequence_model_results.pkl"), "wb") as f:
        pickle.dump(results, f)

    print(results)


if __name__ == "__main__":
    main()
