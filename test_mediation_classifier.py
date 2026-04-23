"""
Meditation inference script.

Primary artifact contract:
- models/meditation_model_bundle.pkl (shared feature pipeline + model(s) + threshold)

Legacy fallback contract:
- meditation_svm_classifier.pkl
- model_feature_names.pkl
- scaler.pkl
"""

from __future__ import annotations

import glob
import os
import pickle
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from meditation_pipeline import FEATURE_NAMES_19, MeditationFeaturePipeline, align_features

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

LEVEL_NAMES = {0: "Baseline", 1: "Relaxed", 2: "Focused", 3: "Deep", -1: "Uncertain"}
LEVEL_COLORS = {
    0: "#e74c3c",
    1: "#f39c12",
    2: "#2ecc71",
    3: "#3498db",
    -1: "#7f8c8d",
}


def resolve_target_file() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1]
    candidates = sorted(glob.glob(os.path.join(SCRIPT_DIR, "meditating.csv")))
    if candidates:
        print(f"[INFO] Auto-detected: {candidates[0]}")
        return candidates[0]
    raise FileNotFoundError("No CSV provided. Usage: python test_mediation_classifier.py <path_to_csv>")


def load_bundle_or_legacy(base_dir: str):
    bundle_candidates = [
        os.path.join(base_dir, "models", "meditation_model_bundle.pkl"),
        os.path.join(base_dir, "meditation_model_bundle.pkl"),
    ]
    for p in bundle_candidates:
        if os.path.exists(p):
            with open(p, "rb") as f:
                bundle = pickle.load(f)
            print(f"[LOAD] Bundle: {p}")
            return "bundle", bundle

    model_path = os.path.join(base_dir, "meditation_svm_classifier.pkl")
    features_path = os.path.join(base_dir, "model_feature_names.pkl")
    scaler_path = os.path.join(base_dir, "scaler.pkl")
    for path, label in [
        (model_path, "Model"),
        (features_path, "Feature names"),
        (scaler_path, "Scaler"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{label} not found: {path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(features_path, "rb") as f:
        feature_names = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    print("[LOAD] Legacy artifacts")
    return "legacy", {"model": model, "feature_names": feature_names, "scaler": scaler}


def predict_with_bundle(bundle: dict, feat_df: pd.DataFrame):
    model_name = bundle.get("selected_model", "svm")
    model = bundle["models"][model_name]
    feature_names = bundle.get("feature_names", FEATURE_NAMES_19)
    reject_thr = float(bundle.get("reject_threshold", 0.5))

    X = align_features(feat_df, feature_names)
    pred = model.predict(X)
    proba = model.predict_proba(X)
    conf = proba.max(axis=1)
    pred = np.where(conf < reject_thr, -1, pred)
    return pred, proba, conf, reject_thr


def predict_with_legacy(legacy: dict, feat_df: pd.DataFrame):
    X = align_features(feat_df, legacy["feature_names"]).values.astype(np.float64)
    X_scaled = legacy["scaler"].transform(X)
    pred = legacy["model"].predict(X_scaled)
    proba = legacy["model"].predict_proba(X_scaled)
    conf = proba.max(axis=1)
    return pred, proba, conf, 0.0


def summarize(pred: np.ndarray, conf: np.ndarray):
    print("\n[PREDICT] Distribution:")
    for lvl in sorted(LEVEL_NAMES):
        mask = pred == lvl
        if not mask.any():
            continue
        n = mask.sum()
        pct = 100 * n / len(pred)
        print(f"  L{lvl:>2} {LEVEL_NAMES[lvl]:10s}: {n:4d} ({pct:5.1f}%)")

    print(f"\n[CONF] mean={conf.mean():.3f} median={np.median(conf):.3f}")


def make_plot(times, pred, conf, results: pd.DataFrame, out_png: str):
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    ax = axes[0]
    for lvl in sorted(LEVEL_NAMES):
        mask = pred == lvl
        if mask.any():
            ax.scatter(
                times[mask],
                np.full(mask.sum(), lvl),
                color=LEVEL_COLORS[lvl],
                s=12,
                alpha=0.85,
                label=f"L{lvl}: {LEVEL_NAMES[lvl]}",
            )
    ax.set_ylabel("Level")
    ax.set_title("Predicted Meditation Level")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=3, loc="upper right")

    ax = axes[1]
    ax.fill_between(times, conf, alpha=0.15, color="purple")
    ax.plot(times, conf, color="purple", linewidth=1.0)
    ax.axhline(0.70, linestyle="--", color="green", linewidth=0.8)
    ax.axhline(0.50, linestyle="--", color="red", linewidth=0.8)
    ax.set_ylabel("Confidence")
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Confidence")
    ax.grid(alpha=0.3)

    ax = axes[2]
    for band in ["Delta", "Theta", "Alpha", "Beta", "Gamma"]:
        col = f"mean_rel_{band}"
        if col in results.columns:
            ax.plot(times, results[col].values, linewidth=1.0, alpha=0.85, label=band)
    ax.set_ylabel("Relative Power")
    ax.set_xlabel("Time (s)")
    ax.set_title("Band Power Over Time")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=5)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main() -> None:
    print("=" * 70)
    print("  MEDITATION CLASSIFIER — INFERENCE")
    print("=" * 70)

    target_file = resolve_target_file()
    if not os.path.exists(target_file):
        raise FileNotFoundError(target_file)

    raw_df = pd.read_csv(target_file, low_memory=False)
    mode, artifacts = load_bundle_or_legacy(SCRIPT_DIR)

    if mode == "bundle":
        feature_pipeline = artifacts.get("feature_pipeline", MeditationFeaturePipeline())
    else:
        feature_pipeline = MeditationFeaturePipeline()

    feat_df, meta_df, stats = feature_pipeline.extract(raw_df, drop_gated_windows=True)
    if feat_df.empty:
        raise RuntimeError("No valid windows after quality/motion gating")

    print(
        f"[FEATURES] windows={stats['n_windows_valid']} skipped_motion={stats['skipped_motion']} "
        f"skipped_quality={stats['skipped_quality']}"
    )

    if mode == "bundle":
        pred, proba, conf, reject_thr = predict_with_bundle(artifacts, feat_df)
    else:
        pred, proba, conf, reject_thr = predict_with_legacy(artifacts, feat_df)

    summarize(pred, conf)
    if reject_thr > 0:
        print(f"[INFO] Uncertainty threshold applied: {reject_thr:.2f}")

    n_classes = proba.shape[1]
    base = os.path.splitext(os.path.basename(target_file))[0]

    feature_names = artifacts.get("feature_names", FEATURE_NAMES_19) if mode == "bundle" else artifacts["feature_names"]
    results = align_features(feat_df, feature_names).copy()
    results.insert(0, "window_time", meta_df["window_time"].values[: len(results)])
    results["predicted_level"] = pred
    results["level_name"] = [LEVEL_NAMES.get(int(p), str(p)) for p in pred]
    results["confidence"] = np.round(conf, 4)
    for i in range(n_classes):
        results[f"prob_class_{i}"] = np.round(proba[:, i], 4)

    out_csv = os.path.join(SCRIPT_DIR, f"{base}_predictions.csv")
    results.to_csv(out_csv, index=False)

    if pd.api.types.is_datetime64_any_dtype(pd.Series(meta_df["window_time"])):
        t0 = pd.Timestamp(meta_df["window_time"].iloc[0])
        times = np.array([(pd.Timestamp(t) - t0).total_seconds() for t in meta_df["window_time"]])
    else:
        times = pd.to_numeric(meta_df["window_time"], errors="coerce").fillna(0).values

    out_png = os.path.join(SCRIPT_DIR, f"{base}_timeline.png")
    make_plot(times, pred, conf, results, out_png)

    print(f"[SAVE] CSV  -> {out_csv}")
    print(f"[SAVE] Plot -> {out_png}")


if __name__ == "__main__":
    main()
