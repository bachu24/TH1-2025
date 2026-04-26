#!/usr/bin/env python3
"""
=================================================================
  Model 2: Random Forest — Group-based Classification
=================================================================
  Run: python train_rf.py
=================================================================
"""
import os, json, pickle, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.feature_selection import SelectKBest, f_classif
warnings.filterwarnings('ignore')

OUT = os.environ.get("OUTPUT_DIR", "/sessions/practical-happy-pasteur/mnt/TH1 Brain Wave")
RS = 42; K_BEST = 200
GROUP_NAMES = {1: "Novice", 2: "Intermediate", 3: "Expert"}

def main():
    print("=" * 60); print("  MODEL 2: Random Forest"); print("=" * 60)
    df = pd.read_csv(os.path.join(OUT, 'processed_dataset.csv'))
    with open(os.path.join(OUT, 'feature_names.pkl'), 'rb') as f: feat_names = pickle.load(f)
    X = df[feat_names].values; y = df['group'].values
    cv = StratifiedKFold(5, shuffle=True, random_state=RS)

    # 5-Fold CV
    accs, baccs, f1s = [], [], []
    all_yt, all_yp = [], []
    for fold_i, (tr, te) in enumerate(cv.split(X, y)):
        sc = StandardScaler(); Xt = sc.fit_transform(X[tr]); Xv = sc.transform(X[te])
        sel = SelectKBest(f_classif, k=K_BEST)
        Xt = sel.fit_transform(Xt, y[tr]); Xv = sel.transform(Xv)
        clf = RandomForestClassifier(n_estimators=500, max_depth=15,
              class_weight='balanced', random_state=RS)
        clf.fit(Xt, y[tr]); yp = clf.predict(Xv)
        a = accuracy_score(y[te], yp); accs.append(a)
        baccs.append(balanced_accuracy_score(y[te], yp))
        f1s.append(f1_score(y[te], yp, average='macro'))
        all_yt.extend(y[te]); all_yp.extend(yp)
        print(f"  Fold {fold_i+1}: acc={a:.4f}")

    cv_acc = np.mean(accs)
    print(f"\n  CV: acc={cv_acc:.4f}+/-{np.std(accs):.4f}, bacc={np.mean(baccs):.4f}, f1={np.mean(f1s):.4f}")
    print(classification_report(all_yt, all_yp,
          target_names=[GROUP_NAMES[i] for i in sorted(np.unique(y))]))

    # Confusion matrix
    labels = sorted(np.unique(y))
    cm = confusion_matrix(all_yt, all_yp, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=[GROUP_NAMES[i] for i in labels]).plot(ax=ax, cmap='Greens')
    ax.set_title(f'Random Forest CV (Acc={cv_acc:.1%})')
    plt.tight_layout(); plt.savefig(os.path.join(OUT, 'rf_cm.png'), dpi=150); plt.close()

    # Feature importance
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    sel = SelectKBest(f_classif, k=K_BEST); Xs = sel.fit_transform(Xs, y)
    sel_mask = sel.get_support()
    sel_names = [feat_names[i] for i in range(len(feat_names)) if sel_mask[i]]
    clf = RandomForestClassifier(n_estimators=500, max_depth=15, class_weight='balanced', random_state=RS)
    clf.fit(Xs, y)
    imp = clf.feature_importances_; sidx = np.argsort(imp)[::-1][:15]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh([sel_names[i][-30:] for i in sidx][::-1], [imp[i] for i in sidx][::-1],
            color=['#e63946' if i < 3 else '#27ae60' for i in range(15)][::-1])
    ax.set_xlabel('MDI'); ax.set_title('RF Feature Importance (Top 15)')
    plt.tight_layout(); plt.savefig(os.path.join(OUT, 'rf_feature_importance.png'), dpi=150); plt.close()

    # Save
    with open(os.path.join(OUT, 'rf_model.pkl'), 'wb') as f: pickle.dump(clf, f)
    with open(os.path.join(OUT, 'rf_scaler.pkl'), 'wb') as f: pickle.dump(sc, f)
    with open(os.path.join(OUT, 'rf_selector.pkl'), 'wb') as f: pickle.dump(sel, f)
    results = {'model': 'Random Forest', 'k_best': K_BEST,
               'cv5_accuracy': round(cv_acc, 4), 'cv5_accuracy_std': round(float(np.std(accs)), 4),
               'cv5_balanced_accuracy': round(float(np.mean(baccs)), 4),
               'cv5_macro_f1': round(float(np.mean(f1s)), 4),
               'top_features': [sel_names[sidx[i]] for i in range(min(10, len(sidx)))]}
    with open(os.path.join(OUT, 'rf_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: rf_model.pkl, rf_results.json, rf_cm.png, rf_feature_importance.png")

if __name__ == "__main__": main()
