from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

MUSE_CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
EPS = 1e-8

FEATURE_NAMES_19 = [
    "mean_rel_Delta",
    "mean_rel_Theta",
    "mean_rel_Alpha",
    "mean_rel_Beta",
    "mean_rel_Gamma",
    "std_rel_Delta",
    "std_rel_Theta",
    "std_rel_Alpha",
    "std_rel_Beta",
    "std_rel_Gamma",
    "ratio_theta_alpha",
    "ratio_alpha_beta",
    "ratio_delta_alpha",
    "ratio_theta_beta",
    "ratio_delta_beta",
    "ratio_gamma_alpha",
    "frontal_alpha_asymm",
    "frontal_theta_mean",
    "occipital_alpha_mean",
]


@dataclass
class ExtractionConfig:
    fs: int = 256
    window_sec: int = 2
    stride_sec: int = 1
    motion_threshold: float = 0.08
    min_quality_fraction: float = 0.70


class MeditationFeaturePipeline:
    def __init__(self, config: ExtractionConfig | None = None):
        self.config = config or ExtractionConfig()

    def detect_schema(self, raw_df: pd.DataFrame) -> Dict[str, Any]:
        band_cols: Dict[str, Dict[str, str]] = {}
        for band in BANDS:
            band_cols[band] = {}
            for ch in MUSE_CHANNELS:
                exact = f"{band}_{ch}"
                if exact in raw_df.columns:
                    band_cols[band][ch] = exact
                else:
                    alts = [
                        c
                        for c in raw_df.columns
                        if band.lower() in c.lower() and ch.lower() in c.lower()
                    ]
                    if alts:
                        band_cols[band][ch] = alts[0]

        imu_cols = [
            c
            for c in ["Accelerometer_X", "Accelerometer_Y", "Accelerometer_Z"]
            if c in raw_df.columns
        ]
        if not imu_cols:
            imu_cols = [
                c
                for c in raw_df.columns
                if "acc" in c.lower() or "accel" in c.lower()
            ]

        time_col = None
        for c in ["TimeStamp", "timestamps", "Timestamp", "time", "Time"]:
            if c in raw_df.columns:
                time_col = c
                break

        hsi_cols = [c for c in raw_df.columns if c.startswith("HSI_")]
        headband_col = "HeadBandOn" if "HeadBandOn" in raw_df.columns else None

        return {
            "band_cols": band_cols,
            "imu_cols": imu_cols,
            "time_col": time_col,
            "hsi_cols": hsi_cols,
            "headband_col": headband_col,
        }

    def estimate_fs(self, raw_df: pd.DataFrame, time_col: str | None) -> int:
        fs = self.config.fs
        if not time_col or len(raw_df) <= 100:
            return fs

        ts = pd.to_datetime(raw_df[time_col], errors="coerce")
        ts = ts.dropna()
        if len(ts) <= 100:
            return fs

        ts_sec = ts.astype(np.int64) / 1e9
        dt = np.diff(ts_sec[:500])
        dt = dt[(dt > 0) & (dt < 1.0)]
        if len(dt) > 10:
            est_fs = int(round(1.0 / np.median(dt)))
            if 50 <= est_fs <= 512:
                return est_fs
        return fs

    def build_quality_mask(
        self,
        raw_df: pd.DataFrame,
        hsi_cols: List[str],
        headband_col: str | None,
    ) -> pd.Series:
        quality_ok = pd.Series(True, index=raw_df.index)

        if headband_col:
            hb = pd.to_numeric(raw_df[headband_col], errors="coerce")
            quality_ok &= hb == 1

        if hsi_cols:
            hsi = raw_df[hsi_cols].apply(pd.to_numeric, errors="coerce")
            hsi_bad = (hsi > 2).any(axis=1)
            quality_ok &= ~hsi_bad

        return quality_ok

    @staticmethod
    def _channel_rel_power(
        win_df: pd.DataFrame,
        band_cols: Dict[str, Dict[str, str]],
        band: str,
        ch: str,
        total_power: float,
    ) -> float:
        if ch not in band_cols.get(band, {}):
            return np.nan
        col = band_cols[band][ch]
        vals = pd.to_numeric(win_df[col], errors="coerce").dropna()
        if len(vals) == 0:
            return np.nan
        return float((10.0 ** vals).mean()) / total_power

    def extract(
        self,
        raw_df: pd.DataFrame,
        drop_gated_windows: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
        raw_df = raw_df.copy()
        raw_df.dropna(how="all", inplace=True)
        raw_df.reset_index(drop=True, inplace=True)

        schema = self.detect_schema(raw_df)
        band_cols = schema["band_cols"]

        all_bp_cols = [c for b in BANDS for c in band_cols[b].values()]
        if not all_bp_cols:
            raise ValueError("No band-power columns found for feature extraction")

        raw_df[all_bp_cols] = raw_df[all_bp_cols].apply(pd.to_numeric, errors="coerce")

        time_col = schema["time_col"]
        if time_col:
            raw_df[time_col] = pd.to_datetime(raw_df[time_col], errors="coerce")

        fs = self.estimate_fs(raw_df, time_col)
        window_samp = fs * self.config.window_sec
        stride_samp = fs * self.config.stride_sec

        quality_ok = self.build_quality_mask(
            raw_df,
            schema["hsi_cols"],
            schema["headband_col"],
        )

        imu_cols = schema["imu_cols"]
        has_imu = len(imu_cols) > 0
        if has_imu:
            raw_df[imu_cols] = raw_df[imu_cols].apply(pd.to_numeric, errors="coerce")

        n_rows = len(raw_df)
        n_win = max(0, (n_rows - window_samp) // stride_samp + 1)

        records: List[Dict[str, float]] = []
        meta_rows: List[Dict[str, Any]] = []
        skipped_motion = 0
        skipped_quality = 0

        for i in range(n_win):
            start = i * stride_samp
            end = start + window_samp
            win = raw_df.iloc[start:end]

            window_quality_ok = quality_ok.iloc[start:end].mean() >= self.config.min_quality_fraction
            if not window_quality_ok and drop_gated_windows:
                skipped_quality += 1
                continue

            motion_ok = True
            if has_imu:
                imu_vals = win[imu_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values
                resultant = np.sqrt((imu_vals**2).sum(axis=1))
                motion_ok = (resultant - resultant.mean()).std() <= self.config.motion_threshold
                if not motion_ok and drop_gated_windows:
                    skipped_motion += 1
                    continue

            bp_linear: Dict[str, np.ndarray] = {}
            for band in BANDS:
                ch_vals = []
                for _, col in band_cols[band].items():
                    vals = pd.to_numeric(win[col], errors="coerce").dropna()
                    if len(vals) > 0:
                        ch_vals.append(float((10.0 ** vals).mean()))
                bp_linear[band] = np.array(ch_vals) if ch_vals else np.array([0.0])

            total_power = sum(bp_linear[b].mean() for b in BANDS) + EPS
            row: Dict[str, float] = {}

            for band in BANDS:
                row[f"mean_rel_{band}"] = float(bp_linear[band].mean()) / total_power
            for band in BANDS:
                vals = bp_linear[band] / total_power
                row[f"std_rel_{band}"] = float(vals.std()) if len(vals) > 1 else 0.0

            row["ratio_theta_alpha"] = row["mean_rel_Theta"] / (row["mean_rel_Alpha"] + EPS)
            row["ratio_alpha_beta"] = row["mean_rel_Alpha"] / (row["mean_rel_Beta"] + EPS)
            row["ratio_delta_alpha"] = row["mean_rel_Delta"] / (row["mean_rel_Alpha"] + EPS)
            row["ratio_theta_beta"] = row["mean_rel_Theta"] / (row["mean_rel_Beta"] + EPS)
            row["ratio_delta_beta"] = row["mean_rel_Delta"] / (row["mean_rel_Beta"] + EPS)
            row["ratio_gamma_alpha"] = row["mean_rel_Gamma"] / (row["mean_rel_Alpha"] + EPS)

            af7_a = self._channel_rel_power(win, band_cols, "Alpha", "AF7", total_power)
            af8_a = self._channel_rel_power(win, band_cols, "Alpha", "AF8", total_power)
            row["frontal_alpha_asymm"] = (
                (af7_a - af8_a) if not np.isnan(af7_a) and not np.isnan(af8_a) else 0.0
            )

            ftheta = [self._channel_rel_power(win, band_cols, "Theta", ch, total_power) for ch in ["AF7", "AF8"]]
            ftheta = [v for v in ftheta if not np.isnan(v)]
            row["frontal_theta_mean"] = float(np.mean(ftheta)) if ftheta else row["mean_rel_Theta"]

            oalpha = [self._channel_rel_power(win, band_cols, "Alpha", ch, total_power) for ch in ["TP9", "TP10"]]
            oalpha = [v for v in oalpha if not np.isnan(v)]
            row["occipital_alpha_mean"] = float(np.mean(oalpha)) if oalpha else row["mean_rel_Alpha"]

            records.append(row)
            meta_rows.append(
                {
                    "window_index": i,
                    "start_idx": start,
                    "end_idx": end,
                    "window_time": raw_df[time_col].iloc[start] if time_col else start / fs,
                    "quality_ok": window_quality_ok,
                    "motion_ok": motion_ok,
                    "fs": fs,
                }
            )

        feat_df = pd.DataFrame(records)
        meta_df = pd.DataFrame(meta_rows)
        stats = {
            "n_rows": n_rows,
            "n_windows_expected": n_win,
            "n_windows_valid": len(feat_df),
            "skipped_motion": skipped_motion,
            "skipped_quality": skipped_quality,
        }
        return feat_df, meta_df, stats


def align_features(feat_df: pd.DataFrame, feature_names: Iterable[str]) -> pd.DataFrame:
    out = feat_df.copy()
    for f in feature_names:
        if f not in out.columns:
            out[f] = 0.0
    out = out[list(feature_names)]
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out
