from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from meditation_pipeline import FEATURE_NAMES_19, MeditationFeaturePipeline, align_features

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "imblearn is required for SMOTE-inside-CV training. Install: pip install imbalanced-learn"
    ) from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train SVM and RF meditation models with subject-wise evaluation")
    p.add_argument("--input", nargs="+", required=True, help="Raw CSV files (one or more)")
    p.add_argument("--label-col", default="label", help="Per-row class label column")
    p.add_argument("--subject-col", default="subject_id", help="Subject ID column")
    p.add_argument("--out-dir", default="models", help="Directory to write artifacts")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--min-coverage", type=float, default=0.7)
    return p.parse_args()


def mode_or_nan(values: pd.Series) -> float:
    vals = values.dropna()
    if len(vals) == 0:
        return np.nan
    m = vals.mode()
    return m.iloc[0] if len(m) else np.nan


def make_window_dataset(
    csv_paths: List[str],
    label_col: str,
    subject_col: str,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    feature_pipeline = MeditationFeaturePipeline()
    all_rows = []
    all_y = []
    all_groups = []

    for path in csv_paths:
        df = pd.read_csv(path, low_memory=False)
        if label_col not in df.columns:
            raise ValueError(f"{path}: missing label column '{label_col}'")

        feat_df, meta_df, stats = feature_pipeline.extract(df, drop_gated_windows=True)
        if feat_df.empty:
            print(f"[WARN] {path}: no valid windows after gating")
            continue

        if subject_col in df.columns:
            subject_value = str(df[subject_col].dropna().iloc[0]) if df[subject_col].notna().any() else os.path.basename(path)
        else:
            subject_value = os.path.splitext(os.path.basename(path))[0]

        y_vals = []
        for _, meta in meta_df.iterrows():
            win_labels = pd.to_numeric(df[label_col].iloc[int(meta.start_idx):int(meta.end_idx)], errors="coerce")
            y_vals.append(mode_or_nan(win_labels))

        y_vals = np.asarray(y_vals)
        keep = np.isfinite(y_vals)
        feat_df = feat_df.loc[keep].reset_index(drop=True)
        y_vals = y_vals[keep].astype(int)

        if len(feat_df) == 0:
            print(f"[WARN] {path}: all labels invalid at window level")
            continue

        feat_df["__subject__"] = subject_value
        all_rows.append(feat_df)
        all_y.append(y_vals)
        all_groups.append(np.array([subject_value] * len(y_vals)))
        print(f"[DATA] {path}: windows={len(y_vals)} valid={stats['n_windows_valid']}")

    if not all_rows:
        raise RuntimeError("No training windows were produced")

    X = pd.concat(all_rows, ignore_index=True)
    groups = X.pop("__subject__").values
    y = np.concatenate(all_y)

    X = align_features(X, FEATURE_NAMES_19)
    return X, y, groups


def build_model_grids(random_state: int) -> Dict[str, Tuple[ImbPipeline, Dict[str, List]]]:
    base_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("smote", SMOTE(random_state=random_state)),
    ]

    svm = ImbPipeline(
        steps=base_steps
        + [
            (
                "clf",
                SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    probability=True,
                    random_state=random_state,
                ),
            )
        ]
    )
    rf = ImbPipeline(
        steps=base_steps
        + [
            (
                "clf",
                RandomForestClassifier(
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            )
        ]
    )

    grids = {
        "svm": (
            svm,
            {
                "clf__C": [0.1, 1, 10],
                "clf__gamma": ["scale", 0.01, 0.001],
            },
        ),
        "rf": (
            rf,
            {
                "clf__n_estimators": [200, 400],
                "clf__max_depth": [None, 12, 24],
                "clf__min_samples_split": [2, 5],
                "clf__min_samples_leaf": [1, 2],
            },
        ),
    }
    return grids


def select_reject_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    conf: np.ndarray,
    min_coverage: float,
) -> float:
    best_thr = 0.5
    best_score = -1.0
    for thr in np.arange(0.4, 0.91, 0.02):
        keep = conf >= thr
        coverage = keep.mean()
        if coverage < min_coverage or keep.sum() == 0:
            continue
        score = f1_score(y_true[keep], y_pred[keep], average="macro")
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr


def evaluate_model(name: str, model, X_val: pd.DataFrame, y_val: np.ndarray, min_coverage: float) -> Dict:
    proba = model.predict_proba(X_val)
    pred = model.predict(X_val)
    conf = proba.max(axis=1)
    thr = select_reject_threshold(y_val, pred, conf, min_coverage=min_coverage)

    uncertain = conf < thr
    accepted = ~uncertain

    metrics = {
        "model": name,
        "macro_f1": float(f1_score(y_val, pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_val, pred)),
        "macro_recall": float(recall_score(y_val, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_val, pred).tolist(),
        "classification_report": classification_report(y_val, pred, zero_division=0),
        "reject_threshold": float(thr),
        "coverage_at_threshold": float(accepted.mean()),
    }

    if accepted.any():
        metrics["macro_f1_accepted"] = float(f1_score(y_val[accepted], pred[accepted], average="macro"))
    else:
        metrics["macro_f1_accepted"] = 0.0

    return metrics


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    X, y, groups = make_window_dataset(args.input, args.label_col, args.subject_col)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise RuntimeError("Need at least 3 unique subjects for subject-wise split/CV")

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.random_state)
    train_idx, val_idx = next(splitter.split(X, y, groups=groups))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    g_train = groups[train_idx]

    n_splits = min(5, len(np.unique(g_train)))
    if n_splits < 2:
        raise RuntimeError("Not enough training subjects for GroupKFold")

    cv = GroupKFold(n_splits=n_splits)
    grids = build_model_grids(args.random_state)

    trained = {}
    evaluations = {}

    for name, (pipe, param_grid) in grids.items():
        print(f"[TRAIN] {name.upper()} with GroupKFold(n_splits={n_splits})")
        search = GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=1,
        )
        search.fit(X_train, y_train, groups=g_train)

        calibrated = CalibratedClassifierCV(
            estimator=search.best_estimator_,
            method="sigmoid",
            cv=3,
        )
        calibrated.fit(X_train, y_train)

        eval_metrics = evaluate_model(name, calibrated, X_val, y_val, min_coverage=args.min_coverage)
        eval_metrics["best_params"] = search.best_params_

        trained[name] = calibrated
        evaluations[name] = eval_metrics
        print(f"[VAL] {name}: macro_f1={eval_metrics['macro_f1']:.4f}, bal_acc={eval_metrics['balanced_accuracy']:.4f}")

    selected = max(evaluations, key=lambda k: evaluations[k]["macro_f1"])
    print(f"[SELECT] best model = {selected}")

    bundle = {
        "feature_pipeline": MeditationFeaturePipeline(),
        "feature_names": FEATURE_NAMES_19,
        "models": trained,
        "selected_model": selected,
        "reject_threshold": evaluations[selected]["reject_threshold"],
        "evaluation": evaluations,
    }

    with open(os.path.join(args.out_dir, "meditation_model_bundle.pkl"), "wb") as f:
        pickle.dump(bundle, f)

    with open(os.path.join(args.out_dir, "training_evaluation.json"), "w", encoding="utf-8") as f:
        json.dump(evaluations, f, indent=2)

    # Legacy-compatible artifacts for existing inference fallback.
    legacy_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_train_scaled = legacy_pipe.fit_transform(X_train)
    svm_legacy = SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=args.random_state)
    svm_legacy.fit(X_train_scaled, y_train)

    with open(os.path.join(args.out_dir, "meditation_svm_classifier.pkl"), "wb") as f:
        pickle.dump(svm_legacy, f)
    with open(os.path.join(args.out_dir, "model_feature_names.pkl"), "wb") as f:
        pickle.dump(FEATURE_NAMES_19, f)
    with open(os.path.join(args.out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(legacy_pipe.named_steps["scaler"], f)

    print("[SAVE] models/meditation_model_bundle.pkl")
    print("[SAVE] models/training_evaluation.json")


if __name__ == "__main__":
    main()
