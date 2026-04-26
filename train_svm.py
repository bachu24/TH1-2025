#!/usr/bin/env python3
"""
=================================================================
  Model 1: SVM (RBF Kernel) — PRIMARY MODEL
  Enhanced channel-agnostic features + RFE + SMOTE + Isotonic Cal.
  Best: RFE=60 features, C=5, gamma=0.005 → LOSO 91.7%
=================================================================
  Run: python train_svm.py
  Requires: processed_dataset.csv from preprocess_data.py
=================================================================
"""
import os, json, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC, LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.feature_selection import RFE
from sklearn.impute import SimpleImputer
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
warnings.filterwarnings('ignore')

OUT = os.environ.get("OUTPUT_DIR",
    "/sessions/practical-happy-pasteur/mnt/TH1 Brain Wave")
RS = 42
N_RFE = 60
SVM_C = 5
SVM_GAMMA = 0.005
GROUP_NAMES = {1: "Novice", 2: "Intermediate", 3: "Expert"}


def main():
    print("=" * 60)
    print("  MODEL 1: SVM (RBF) — Enhanced Channel-Agnostic")
    print("=" * 60)

    # Load data
    df = pd.read_csv(os.path.join(OUT, 'processed_dataset.csv'))
    with open(os.path.join(OUT, 'feature_names.pkl'), 'rb') as f:
        feat_names = pickle.load(f)
    X = df[feat_names].values
    y = df['group'].values
    print(f"  Data: {X.shape[0]} subjects, {X.shape[1]} features")
    print(f"  Config: RFE→{N_RFE}, C={SVM_C}, gamma={SVM_GAMMA}, kernel=rbf")

    # Standardize + RFE
    print("\n[1] RFE Feature Selection...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    rfe = RFE(
        LinearSVC(C=1.0, class_weight='balanced', random_state=RS, max_iter=5000),
        n_features_to_select=N_RFE, step=20)
    rfe.fit(X_scaled, y)
    X_rfe = X_scaled[:, rfe.support_]
    rfe_feat_names = [feat_names[i] for i in range(len(feat_names)) if rfe.support_[i]]
    print(f"  Selected {sum(rfe.support_)} features from {X.shape[1]}")

    # ── LOSO with SMOTE + Calibration ──
    print("\n[2] Leave-One-Subject-Out CV (60 folds)...")
    loso = LeaveOneOut()
    loso_preds, loso_probs = [], []
    for tr, te in loso.split(X_rfe):
        X_tr, X_te = X_rfe[tr], X_rfe[te]
        y_tr = y[tr]
        # SMOTE inside fold
        counts = np.bincount(y_tr)
        min_c = min(counts[counts > 0])
        if min_c >= 4:
            smote = SMOTE(random_state=RS, k_neighbors=min(3, min_c - 1))
            X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
        # Calibrated SVM
        base = SVC(C=SVM_C, gamma=SVM_GAMMA, kernel='rbf',
                   class_weight='balanced', random_state=RS)
        cal = CalibratedClassifierCV(base, cv=3, method='isotonic')
        cal.fit(X_tr, y_tr)
        loso_preds.append(cal.predict(X_te)[0])
        loso_probs.append(cal.predict_proba(X_te)[0])

    loso_preds = np.array(loso_preds)
    loso_probs = np.array(loso_probs)
    loso_acc = accuracy_score(y, loso_preds)
    loso_bacc = balanced_accuracy_score(y, loso_preds)
    loso_f1 = f1_score(y, loso_preds, average='macro')

    print(f"\n  LOSO Results:")
    print(f"    Accuracy:          {loso_acc:.4f} ({loso_acc*100:.1f}%)")
    print(f"    Balanced Accuracy: {loso_bacc:.4f}")
    print(f"    Macro-F1:          {loso_f1:.4f}")
    print(classification_report(y, loso_preds,
          target_names=[GROUP_NAMES[i] for i in sorted(np.unique(y))]))

    correct = y == loso_preds
    conf_correct = [max(p) for p, c in zip(loso_probs, correct) if c]
    conf_wrong = [max(p) for p, c in zip(loso_probs, correct) if not c]
    print(f"  Confidence (correct): mean={np.mean(conf_correct):.3f}")
    if conf_wrong:
        print(f"  Confidence (wrong):   mean={np.mean(conf_wrong):.3f}")

    # ── 5-Fold CV ──
    print("\n[3] 5-Fold Stratified CV...")
    cv = StratifiedKFold(5, shuffle=True, random_state=RS)
    cv_accs, cv_f1s = [], []
    cv_yt, cv_yp = [], []
    for fold_i, (tr, te) in enumerate(cv.split(X_rfe, y)):
        X_tr, X_te = X_rfe[tr], X_rfe[te]
        y_tr = y[tr]
        counts = np.bincount(y_tr); min_c = min(counts[counts > 0])
        if min_c >= 4:
            smote = SMOTE(random_state=RS, k_neighbors=min(3, min_c - 1))
            X_tr, y_tr = smote.fit_resample(X_tr, y_tr)
        clf = SVC(C=SVM_C, gamma=SVM_GAMMA, kernel='rbf',
                  class_weight='balanced', random_state=RS)
        clf.fit(X_tr, y_tr)
        yp = clf.predict(X_te)
        a = accuracy_score(y[te], yp)
        cv_accs.append(a); cv_f1s.append(f1_score(y[te], yp, average='macro'))
        cv_yt.extend(y[te]); cv_yp.extend(yp)
        print(f"    Fold {fold_i+1}: acc={a:.4f}")
    print(f"  5-Fold: acc={np.mean(cv_accs):.4f}+/-{np.std(cv_accs):.4f}")

    # ── Confusion matrices ──
    print("\n[4] Figures...")
    labels = sorted(np.unique(y))
    for title, yt, yp, suffix in [
        (f"SVM LOSO (Acc={loso_acc:.1%}, F1={loso_f1:.3f})", y, loso_preds, "loso"),
        (f"SVM 5-Fold CV (Acc={np.mean(cv_accs):.1%})", cv_yt, cv_yp, "cv"),
    ]:
        cm = confusion_matrix(yt, yp, labels=labels)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay(cm, display_labels=[GROUP_NAMES[i] for i in labels]).plot(
            ax=ax, cmap='Blues')
        ax.set_title(title, fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, f'svm_ca_{suffix}.png'), dpi=150)
        plt.close()

    # Feature importance
    lin = LinearSVC(C=1.0, class_weight='balanced', random_state=RS, max_iter=5000)
    lin.fit(X_rfe, y)
    coef = np.abs(lin.coef_).mean(axis=0)
    sidx = np.argsort(coef)[::-1][:20]
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [rfe_feat_names[i][-40:] for i in sidx]
    colors = ['#c0392b' if i < 5 else '#2980b9' for i in range(20)]
    ax.barh(names[::-1], [coef[i] for i in sidx][::-1], color=colors[::-1])
    ax.set_xlabel('Mean |Coefficient| (Linear SVM)')
    ax.set_title('Enhanced SVM — Top 20 RFE-Selected Features')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'svm_ca_features.png'), dpi=150)
    plt.close()

    # ── Save model ──
    print("\n[5] Saving...")
    smote_final = SMOTE(random_state=RS, k_neighbors=3)
    X_sm, y_sm = smote_final.fit_resample(X_rfe, y)
    final_base = SVC(C=SVM_C, gamma=SVM_GAMMA, kernel='rbf',
                     class_weight='balanced', random_state=RS)
    final_model = CalibratedClassifierCV(final_base, cv=3, method='isotonic')
    final_model.fit(X_sm, y_sm)

    with open(os.path.join(OUT, 'svm_ca_model.pkl'), 'wb') as f:
        pickle.dump(final_model, f)
    with open(os.path.join(OUT, 'svm_ca_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(OUT, 'svm_ca_rfe.pkl'), 'wb') as f:
        pickle.dump(rfe, f)
    with open(os.path.join(OUT, 'svm_ca_feat_names.pkl'), 'wb') as f:
        with open(os.path.join(OUT, 'ca_window_feat_names.pkl'), 'rb') as f2:
            pickle.dump(pickle.load(f2), f)

    results = {
        'model': 'SVM Enhanced Channel-Agnostic',
        'hyperparameters': {'C': SVM_C, 'gamma': SVM_GAMMA, 'kernel': 'rbf', 'n_rfe': N_RFE},
        'loso_accuracy': round(float(loso_acc), 4),
        'loso_balanced_accuracy': round(float(loso_bacc), 4),
        'loso_macro_f1': round(float(loso_f1), 4),
        'cv5_accuracy': round(float(np.mean(cv_accs)), 4),
        'cv5_accuracy_std': round(float(np.std(cv_accs)), 4),
        'confidence_correct_mean': round(float(np.mean(conf_correct)), 3),
        'n_features_total': int(X.shape[1]),
        'n_features_rfe': N_RFE,
        'top_features': rfe_feat_names[:20],
        'classification_report_loso': classification_report(
            y, loso_preds, output_dict=True,
            target_names=[GROUP_NAMES[i] for i in sorted(np.unique(y))]),
    }
    with open(os.path.join(OUT, 'svm_ca_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"  SVM COMPLETE: LOSO={loso_acc:.1%}, F1={loso_f1:.3f}")
    print(f"  Confidence (correct): {np.mean(conf_correct):.1%}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
