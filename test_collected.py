#!/usr/bin/env python3
"""
=================================================================
  Test Enhanced SVM on Collected Data (SMS@19nov68)
  Uses: sliding-window features, RFE, isotonic calibration
=================================================================
  Run: python test_collected.py
  Input:  SMS@19nov68/*.zip (Muse 2 recordings)
  Output: sms_ca_predictions.csv
  Requires: svm_ca_model.pkl, svm_ca_scaler.pkl, svm_ca_rfe.pkl,
            svm_ca_feat_names.pkl (from train_svm.py)
=================================================================
"""
import os, json, pickle, glob, zipfile, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

OUT = os.environ.get("OUTPUT_DIR",
    "/sessions/practical-happy-pasteur/mnt/TH1 Brain Wave")
BANDS = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]
GN = {1: "Novice", 2: "Intermediate", 3: "Expert"}
eps = 1e-8


def extract_muse_window_features(mdf, start, end, feat_names):
    """Extract enhanced channel-agnostic features from a Muse 2 window.
    Maps 4-channel Muse data to the same feature space as 64-channel training.
    """
    feat = {}
    mb = {}
    for b in BANDS:
        cols = [c for c in mdf.columns if c.startswith(f'{b}_')]
        if cols:
            mb[b] = mdf[cols].values[start:end]

    if not mb:
        return None

    total = eps
    for b in BANDS:
        if b in mb:
            vals = mb[b].flatten()
            vals = vals[~np.isnan(vals)]
            total += np.mean(vals) if len(vals) > 0 else 0

    for b in BANDS:
        for prefix, data_src in [(b, mb.get(b)), (f'{b}/all', None)]:
            if data_src is not None:
                vals = data_src.flatten()
                vals = vals[~np.isnan(vals)]
            elif b in mb:
                vals = (mb[b] / total).flatten()
                vals = vals[~np.isnan(vals)]
            else:
                continue

            if len(vals) == 0:
                continue

            feat[f'{prefix}_g_mean'] = np.mean(vals)
            feat[f'{prefix}_g_std'] = np.std(vals)
            feat[f'{prefix}_g_med'] = np.median(vals)
            feat[f'{prefix}_g_skew'] = float(pd.Series(vals).skew()) if len(vals) > 2 else 0
            feat[f'{prefix}_g_kurt'] = float(pd.Series(vals).kurtosis()) if len(vals) > 3 else 0
            feat[f'{prefix}_g_iqr'] = np.percentile(vals, 75) - np.percentile(vals, 25)
            feat[f'{prefix}_g_cv'] = np.std(vals) / (np.mean(vals) + eps)
            feat[f'{prefix}_g_p10'] = np.percentile(vals, 10)
            feat[f'{prefix}_g_p90'] = np.percentile(vals, 90)

            # Regional: AF7/AF8=frontal, TP9/TP10=temporal
            for reg, mcols in [('frontal', [f'{b}_AF7', f'{b}_AF8']),
                               ('temporal', [f'{b}_TP9', f'{b}_TP10'])]:
                rc = [c for c in mcols if c in mdf.columns]
                if rc:
                    rv = mdf[rc].values[start:end].flatten()
                    rv = rv[~np.isnan(rv)]
                    if len(rv) > 0:
                        feat[f'{prefix}_{reg}_mean'] = np.mean(rv)
                        feat[f'{prefix}_{reg}_std'] = np.std(rv)
                        feat[f'{prefix}_{reg}_cv'] = np.std(rv) / (np.mean(rv) + eps)

    # Ratios
    rm = {b: feat.get(f'{b}/all_g_mean', 0) for b in BANDS}
    feat['r_ta'] = rm['Theta'] / (rm['Alpha'] + eps)
    feat['r_ab'] = rm['Alpha'] / (rm['Beta'] + eps)
    feat['r_tb'] = rm['Theta'] / (rm['Beta'] + eps)
    feat['r_db'] = rm['Delta'] / (rm['Beta'] + eps)
    feat['r_ga'] = rm['Gamma'] / (rm['Alpha'] + eps)
    feat['r_da'] = rm['Delta'] / (rm['Alpha'] + eps)
    feat['r_tg'] = rm['Theta'] / (rm['Gamma'] + eps)
    feat['r_ag'] = rm['Alpha'] / (rm['Gamma'] + eps)

    # Spectral
    bm = {b: feat.get(f'{b}_g_mean', 0) for b in BANDS}
    centers = [2.5, 6, 10, 21, 37.5]
    tp = sum(bm.values()) + eps
    feat['spec_centroid'] = sum(c * bm[b] / tp for c, b in zip(centers, BANDS))
    feat['spec_spread'] = np.sqrt(
        sum(((c - feat['spec_centroid'])**2) * bm[b] / tp for c, b in zip(centers, BANDS)))

    # Asymmetry
    a7 = mdf.get('Alpha_AF7', pd.Series([0])).dropna().values[start:end]
    a8 = mdf.get('Alpha_AF8', pd.Series([0])).dropna().values[start:end]
    feat['faa'] = np.mean(a7) - np.mean(a8) if len(a7) > 0 and len(a8) > 0 else 0
    t7 = mdf.get('Theta_AF7', pd.Series([0])).dropna().values[start:end]
    t8 = mdf.get('Theta_AF8', pd.Series([0])).dropna().values[start:end]
    feat['fta'] = np.mean(t7) - np.mean(t8) if len(t7) > 0 and len(t8) > 0 else 0

    return [feat.get(fn, 0.0) for fn in feat_names]


def main():
    print("=" * 60)
    print("  TESTING ON COLLECTED DATA (Enhanced CA-SVM)")
    print("=" * 60)

    # Load artifacts
    with open(f'{OUT}/svm_ca_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(f'{OUT}/svm_ca_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(f'{OUT}/svm_ca_rfe.pkl', 'rb') as f:
        rfe = pickle.load(f)
    with open(f'{OUT}/svm_ca_feat_names.pkl', 'rb') as f:
        ca_feat_names = pickle.load(f)
    rfe_mask = rfe.support_
    n_ca = len(ca_feat_names)
    print(f"  CA features/window: {n_ca}, RFE features: {sum(rfe_mask)}")

    sms_dir = f'{OUT}/SMS@19nov68'
    zips = sorted(glob.glob(f'{sms_dir}/*.zip'))
    print(f"  Found {len(zips)} recordings")

    preds = []
    for zp in zips:
        sid = os.path.basename(zp).split('.')[0]
        try:
            with zipfile.ZipFile(zp, 'r') as z:
                cn = [n for n in z.namelist() if n.endswith('.csv')][0]
                with z.open(cn) as cf:
                    mdf = pd.read_csv(cf)
        except:
            continue

        # Check we have all bands
        has_bands = sum(1 for b in BANDS
                       if any(c.startswith(f'{b}_') for c in mdf.columns))
        if has_bands < 5:
            continue
        ml = len(mdf)
        if ml < 30:
            continue

        # Create 6 sliding windows
        ws = ml // 6 if ml >= 60 else ml
        nw = min(6, ml // max(ws, 1))
        if nw < 2:
            nw = 2
            ws = ml // 2

        window_feats = []
        for wi in range(nw):
            st = wi * ws
            en = min(st + ws, ml)
            wf = extract_muse_window_features(mdf, st, en, ca_feat_names)
            if wf is not None:
                window_feats.append(wf)

        if len(window_feats) < 2:
            continue

        # 6 temporal aggregations (matching training)
        mat = np.array(window_feats, dtype=float)
        mat = np.nan_to_num(mat)
        if mat.shape[0] < 6:
            mat = np.vstack([mat, np.tile(mat[-1:], (6 - mat.shape[0], 1))])

        f_mean = mat.mean(axis=0)
        f_std = mat.std(axis=0)
        f_min = mat.min(axis=0)
        f_max = mat.max(axis=0)
        f_range = f_max - f_min
        slopes = np.zeros(n_ca)
        x_t = np.arange(mat.shape[0])
        for col in range(n_ca):
            if np.std(mat[:, col]) > 1e-10:
                slopes[col] = np.polyfit(x_t, mat[:, col], 1)[0]

        fvec = np.concatenate([f_mean, f_std, slopes, f_min, f_max, f_range]).reshape(1, -1)

        # Scale → RFE → Predict
        Xs = scaler.transform(fvec)
        Xs_rfe = Xs[:, rfe_mask]
        pred = model.predict(Xs_rfe)[0]
        prob = model.predict_proba(Xs_rfe)[0]

        preds.append({
            'subject': sid,
            'prediction': GN[pred],
            'confidence': round(float(max(prob)), 3),
            'prob_novice': round(float(prob[0]), 3),
            'prob_intermediate': round(float(prob[1]), 3),
            'prob_expert': round(float(prob[2]), 3),
            'n_samples': ml,
            'n_windows': nw,
        })

    pdf = pd.DataFrame(preds)
    pdf.to_csv(f'{OUT}/sms_ca_predictions.csv', index=False)

    vc = pdf['prediction'].value_counts()
    print(f"\n  Results ({len(preds)} subjects):")
    for v, c in vc.items():
        print(f"    {v}: {c} ({c / len(preds) * 100:.1f}%)")

    print(f"\n  Sample predictions:")
    for d in preds[:4]:
        print(f"    {d['subject']}: {d['prediction']} "
              f"(conf={d['confidence']}, N={d['prob_novice']}, "
              f"I={d['prob_intermediate']}, E={d['prob_expert']})")

    print(f"\n  Saved: sms_ca_predictions.csv")


if __name__ == "__main__":
    main()
