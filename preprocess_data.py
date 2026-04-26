#!/usr/bin/env python3
"""
=================================================================
  Step 0: Enhanced Data Preprocessing Pipeline
  Labels: Group-based (Group1=Novice, Group2=Intermediate, Group3=Expert)
  Features: 252 channel-agnostic per window × 6 aggregations = 1512/subject
=================================================================
  Run: python preprocess_data.py
  Output: processed_dataset.csv, feature_names.pkl
=================================================================
"""
import os, pickle, json, warnings
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
warnings.filterwarnings('ignore')

DATA_ROOT = os.environ.get("EEG_DATA_ROOT",
    "/sessions/practical-happy-pasteur/mnt/EEG absolute and relative powers during mindfulness meditation Data from Thai Buddhist monks")
OUT = os.environ.get("OUTPUT_DIR",
    "/sessions/practical-happy-pasteur/mnt/TH1 Brain Wave")

BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
ALL_BANDS = BANDS + ["Delta/all", "Theta/all", "Alpha/all", "Beta/all", "Gamma/all"]
META_COLS = ["File", "Freq_int_name", "Freq_interval"]
GROUPS = {"Group1": 1, "Group2": 2, "Group3": 3}
GROUP_NAMES = {1: "Novice", 2: "Intermediate", 3: "Expert"}
REGIONS = {
    'frontal': ['AF3','AF4','F3','F4','F7','F8','FP1','FP2','FPZ','FZ',
                'FC1','FC2','FC3','FC4','FC5','FC6','FCZ','FT7','FT8','F1','F2','F5','F6'],
    'central': ['C1','C2','C3','C4','C5','C6','CZ'],
    'parietal': ['P1','P2','P3','P4','P5','P6','P7','P8','PZ',
                 'CP1','CP2','CP3','CP4','CP5','CP6','CPZ'],
    'temporal': ['T7','T8','TP7','TP8'],
    'occipital': ['O1','O2','OZ','PO3','PO4','PO5','PO6','PO7','PO8','POZ'],
}
eps = 1e-8


def extract_window_features(db_rows, electrodes):
    """Extract 252 channel-agnostic features from one time window."""
    feat = {}
    for _, row in db_rows.iterrows():
        band = row['Freq_int_name']
        vals = np.array([row[e] for e in electrodes if pd.notna(row[e])])
        if len(vals) == 0:
            continue
        # Global statistics (9 per band × 10 bands = 90)
        feat[f'{band}_g_mean'] = np.mean(vals)
        feat[f'{band}_g_std'] = np.std(vals)
        feat[f'{band}_g_med'] = np.median(vals)
        feat[f'{band}_g_skew'] = float(pd.Series(vals).skew()) if len(vals) > 2 else 0
        feat[f'{band}_g_kurt'] = float(pd.Series(vals).kurtosis()) if len(vals) > 3 else 0
        feat[f'{band}_g_iqr'] = np.percentile(vals, 75) - np.percentile(vals, 25)
        feat[f'{band}_g_cv'] = np.std(vals) / (np.mean(vals) + eps)
        feat[f'{band}_g_p10'] = np.percentile(vals, 10)
        feat[f'{band}_g_p90'] = np.percentile(vals, 90)
        # Regional statistics (3 per region × 5 regions × 10 bands = 150)
        for rn, re in REGIONS.items():
            rv = [row[e] for e in electrodes
                  if e.upper() in [r.upper() for r in re] and pd.notna(row[e])]
            if rv:
                feat[f'{band}_{rn}_mean'] = np.mean(rv)
                feat[f'{band}_{rn}_std'] = np.std(rv)
                feat[f'{band}_{rn}_cv'] = np.std(rv) / (np.mean(rv) + eps)

    # Band-power ratios (8 features)
    rm = {b: feat.get(f'{b}/all_g_mean', 0) for b in BANDS}
    feat['r_ta'] = rm['Theta'] / (rm['Alpha'] + eps)
    feat['r_ab'] = rm['Alpha'] / (rm['Beta'] + eps)
    feat['r_tb'] = rm['Theta'] / (rm['Beta'] + eps)
    feat['r_db'] = rm['Delta'] / (rm['Beta'] + eps)
    feat['r_ga'] = rm['Gamma'] / (rm['Alpha'] + eps)
    feat['r_da'] = rm['Delta'] / (rm['Alpha'] + eps)
    feat['r_tg'] = rm['Theta'] / (rm['Gamma'] + eps)
    feat['r_ag'] = rm['Alpha'] / (rm['Gamma'] + eps)

    # Spectral centroid and spread (2 features)
    bm = {b: feat.get(f'{b}_g_mean', 0) for b in BANDS}
    centers = [2.5, 6, 10, 21, 37.5]
    tp = sum(bm.values()) + eps
    feat['spec_centroid'] = sum(c * bm[b] / tp for c, b in zip(centers, BANDS))
    feat['spec_spread'] = np.sqrt(
        sum(((c - feat['spec_centroid'])**2) * bm[b] / tp for c, b in zip(centers, BANDS)))

    # Frontal alpha asymmetry (1 feature)
    db_a = db_rows[db_rows['Freq_int_name'] == 'Alpha/all']
    if len(db_a) > 0:
        ar = db_a.iloc[0]
        la = [ar[e] for e in electrodes if e.upper() in ['AF3', 'F3', 'F7'] and pd.notna(ar[e])]
        ra = [ar[e] for e in electrodes if e.upper() in ['AF4', 'F4', 'F8'] and pd.notna(ar[e])]
        feat['faa'] = np.mean(la) - np.mean(ra) if la and ra else 0

    # Frontal theta asymmetry (1 feature)
    db_t = db_rows[db_rows['Freq_int_name'] == 'Theta/all']
    if len(db_t) > 0:
        tr = db_t.iloc[0]
        lt = [tr[e] for e in electrodes if e.upper() in ['AF3', 'F3', 'F7'] and pd.notna(tr[e])]
        rt = [tr[e] for e in electrodes if e.upper() in ['AF4', 'F4', 'F8'] and pd.notna(tr[e])]
        feat['fta'] = np.mean(lt) - np.mean(rt) if lt and rt else 0

    return feat


def main():
    print("=" * 60)
    print("  PREPROCESSING: Enhanced Channel-Agnostic Features")
    print("  Group1=Novice | Group2=Intermediate | Group3=Expert")
    print("=" * 60)

    # 1. Load per-window features
    subj_windows = {}
    subj_group = {}
    ca_feat_names = None

    for g_name, g_label in GROUPS.items():
        gp = os.path.join(DATA_ROOT, g_name)
        if not os.path.isdir(gp):
            continue
        for s in sorted(os.listdir(gp)):
            sp = os.path.join(gp, s)
            if not os.path.isdir(sp):
                continue
            for f in sorted(os.listdir(sp)):
                if not f.endswith('.xlsx'):
                    continue
                df = pd.read_excel(os.path.join(sp, f), engine='openpyxl')
                elec = [c for c in df.columns if c not in META_COLS]
                db = df[df['Freq_int_name'].isin(ALL_BANDS)]
                feat = extract_window_features(db, elec)
                if ca_feat_names is None:
                    ca_feat_names = sorted(feat.keys())
                if s not in subj_windows:
                    subj_windows[s] = []
                subj_windows[s].append([feat.get(fn, 0.0) for fn in ca_feat_names])
                subj_group[s] = g_label

    n_ca = len(ca_feat_names)
    print(f"  Channel-agnostic features per window: {n_ca}")
    print(f"  Subjects loaded: {len(subj_windows)}")

    # 2. Subject-level: 6 aggregations (mean + std + slope + min + max + range)
    subjects, labels, feats = [], [], []
    for s, windows in subj_windows.items():
        mat = np.array(windows, dtype=float)
        mat = np.nan_to_num(mat)
        if mat.shape[0] < 6:
            mat = np.vstack([mat, np.tile(mat[-1:], (6 - mat.shape[0], 1))])
        elif mat.shape[0] > 6:
            mat = mat[:6]

        f_mean = mat.mean(axis=0)
        f_std = mat.std(axis=0)
        f_min = mat.min(axis=0)
        f_max = mat.max(axis=0)
        f_range = f_max - f_min
        slopes = np.zeros(n_ca)
        x = np.arange(mat.shape[0])
        for col in range(n_ca):
            if np.std(mat[:, col]) > 1e-10:
                slopes[col] = np.polyfit(x, mat[:, col], 1)[0]

        subjects.append(s)
        labels.append(subj_group[s])
        feats.append(np.concatenate([f_mean, f_std, slopes, f_min, f_max, f_range]))

    X = np.array(feats)
    y = np.array(labels)
    X = SimpleImputer(strategy='mean').fit_transform(X)

    agg_suffixes = ['_wmean', '_wstd', '_wslope', '_wmin', '_wmax', '_wrange']
    full_feat_names = []
    for suf in agg_suffixes:
        for fn in ca_feat_names:
            full_feat_names.append(fn + suf)

    print(f"\n  Feature matrix: {X.shape} ({n_ca} per window × 6 aggregations)")
    print(f"  Groups: Novice={sum(y==1)}, Intermediate={sum(y==2)}, Expert={sum(y==3)}")

    # 3. Save
    out_df = pd.DataFrame(X, columns=full_feat_names)
    out_df['subject'] = subjects
    out_df['group'] = y
    out_df['group_name'] = [GROUP_NAMES[g] for g in y]
    out_df.to_csv(os.path.join(OUT, 'processed_dataset.csv'), index=False)

    with open(os.path.join(OUT, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(full_feat_names, f)
    with open(os.path.join(OUT, 'ca_window_feat_names.pkl'), 'wb') as f:
        pickle.dump(ca_feat_names, f)

    info = {
        'n_subjects': len(subjects),
        'n_features_per_window': n_ca,
        'n_aggregations': 6,
        'aggregations': 'mean, std, slope, min, max, range',
        'n_features_total': len(full_feat_names),
        'labels': 'Group-based: Group1=Novice, Group2=Intermediate, Group3=Expert',
        'class_distribution': {GROUP_NAMES[i]: int(sum(y == i)) for i in [1, 2, 3]},
    }
    with open(os.path.join(OUT, 'preprocessing_info.json'), 'w') as f:
        json.dump(info, f, indent=2)

    print(f"  Saved: processed_dataset.csv, feature_names.pkl, ca_window_feat_names.pkl")
    print("=" * 60)


if __name__ == "__main__":
    main()
