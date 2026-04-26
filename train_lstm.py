#!/usr/bin/env python3
"""
=================================================================
  Model 3: LSTM — Group-based, Sequence of 6 Time Windows
=================================================================
  Run: python train_lstm.py
=================================================================
"""
import os, json, pickle, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

OUT = os.environ.get("OUTPUT_DIR", "/sessions/practical-happy-pasteur/mnt/TH1 Brain Wave")
DATA_ROOT = os.environ.get("EEG_DATA_ROOT",
    "/sessions/practical-happy-pasteur/mnt/EEG absolute and relative powers during mindfulness meditation Data from Thai Buddhist monks")
RS = 42; K_FEAT = 50; MAX_WIN = 6
GROUP_NAMES = {1: "Novice", 2: "Intermediate", 3: "Expert"}

def load_sequences():
    """Load per-subject sequences of 6 windows, select top features."""
    ALL_BANDS = ["Delta","Theta","Alpha","Beta","Gamma",
                 "Delta/all","Theta/all","Alpha/all","Beta/all","Gamma/all"]
    META = ["File","Freq_int_name","Freq_interval"]
    GROUPS = {"Group1": 1, "Group2": 2, "Group3": 3}

    ak = None; subj_seqs = {}; subj_labels = {}
    for g_name, g_label in GROUPS.items():
        gp = os.path.join(DATA_ROOT, g_name)
        if not os.path.isdir(gp): continue
        for s in sorted(os.listdir(gp)):
            sp = os.path.join(gp, s)
            if not os.path.isdir(sp): continue
            vecs = []
            for f in sorted(os.listdir(sp)):
                if not f.endswith('.xlsx'): continue
                df = pd.read_excel(os.path.join(sp, f), engine='openpyxl')
                el = [c for c in df.columns if c not in META]
                db = df[df['Freq_int_name'].isin(ALL_BANDS)]
                fd = {}
                for _, row in db.iterrows():
                    for e in el: fd[f'{e}_{row["Freq_int_name"]}'] = row[e]
                if ak is None: ak = sorted(fd.keys())
                vecs.append([fd.get(k, 0.0) for k in ak])
            if vecs:
                mat = np.nan_to_num(np.array(vecs, dtype=float))
                if mat.shape[0] < MAX_WIN:
                    mat = np.vstack([mat, np.tile(mat[-1:], (MAX_WIN - mat.shape[0], 1))])
                elif mat.shape[0] > MAX_WIN:
                    mat = mat[:MAX_WIN]
                subj_seqs[s] = mat
                subj_labels[s] = g_label

    # Feature selection on pooled windows
    all_mats = np.vstack(list(subj_seqs.values()))
    all_y = np.concatenate([np.full(MAX_WIN, subj_labels[s]) for s in subj_seqs])
    sc = StandardScaler(); ams = sc.fit_transform(all_mats)
    sel = SelectKBest(f_classif, k=K_FEAT); sel.fit(ams, all_y)
    mask = sel.get_support()

    subs = list(subj_seqs.keys())
    X = np.stack([sc.transform(subj_seqs[s])[:, mask] for s in subs])
    y = np.array([subj_labels[s] for s in subs])
    return X, y, subs, sc, mask

def main():
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.utils import to_categorical
    tf.random.set_seed(RS)

    print("=" * 60); print("  MODEL 3: LSTM"); print("=" * 60)
    X, y, subs, sc, mask = load_sequences()
    print(f"  Sequences: {X.shape} (subjects, windows, features)")

    cv = StratifiedKFold(5, shuffle=True, random_state=RS)
    labels = sorted(np.unique(y)); nc = len(labels)
    lm = {v: i for i, v in enumerate(labels)}
    accs = []; all_yt = []; all_yp = []

    for fold_i, (tr, te) in enumerate(cv.split(range(len(subs)), y)):
        ym = np.array([lm[v] for v in y[tr]])
        cc = np.bincount(ym, minlength=nc)
        cw = {c: len(ym) / (nc * cc[c]) if cc[c] > 0 else 1.0 for c in range(nc)}
        model = Sequential([
            Input(shape=(MAX_WIN, K_FEAT)),
            LSTM(32), BatchNormalization(), Dropout(0.3),
            Dense(16, activation='relu'), Dropout(0.2),
            Dense(nc, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(X[tr], to_categorical(ym, nc), epochs=50, batch_size=8,
                  class_weight=cw, verbose=0,
                  callbacks=[EarlyStopping(patience=8, restore_best_weights=True)])
        yp = np.argmax(model.predict(X[te], verbose=0), axis=1)
        inv = {v: k for k, v in lm.items()}
        ypo = [inv[p] for p in yp]
        all_yt.extend(y[te].tolist()); all_yp.extend(ypo)
        a = accuracy_score(y[te], ypo); accs.append(a)
        print(f"  Fold {fold_i+1}: acc={a:.4f}")

    cv_acc = np.mean(accs)
    print(f"\n  CV: acc={cv_acc:.4f}+/-{np.std(accs):.4f}")
    print(classification_report(all_yt, all_yp,
          target_names=[GROUP_NAMES[i] for i in labels]))

    # Confusion matrix
    cm = confusion_matrix(all_yt, all_yp, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=[GROUP_NAMES[i] for i in labels]).plot(ax=ax, cmap='Oranges')
    ax.set_title(f'LSTM CV (Acc={cv_acc:.1%})'); plt.tight_layout()
    plt.savefig(os.path.join(OUT, 'lstm_cm.png'), dpi=150); plt.close()

    # Training curves (full dataset)
    ym_all = np.array([lm[v] for v in y])
    cc = np.bincount(ym_all, minlength=nc)
    cw = {c: len(ym_all)/(nc*cc[c]) for c in range(nc)}
    model = Sequential([Input(shape=(MAX_WIN, K_FEAT)),
        LSTM(32), BatchNormalization(), Dropout(0.3),
        Dense(16, activation='relu'), Dropout(0.2), Dense(nc, activation='softmax')])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    h = model.fit(X, to_categorical(ym_all, nc), epochs=50, batch_size=8,
                  validation_split=0.2, class_weight=cw, verbose=0,
                  callbacks=[EarlyStopping(patience=8, restore_best_weights=True)])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
    a1.plot(h.history['loss'], label='Train'); a1.plot(h.history['val_loss'], label='Val')
    a1.set_title('LSTM Loss'); a1.set_xlabel('Epoch'); a1.legend()
    a2.plot(h.history['accuracy'], label='Train'); a2.plot(h.history['val_accuracy'], label='Val')
    a2.set_title('LSTM Accuracy'); a2.set_xlabel('Epoch'); a2.legend()
    plt.tight_layout(); plt.savefig(os.path.join(OUT, 'lstm_curves.png'), dpi=150); plt.close()

    results = {'model': 'LSTM', 'cv5_accuracy': round(cv_acc, 4),
               'cv5_accuracy_std': round(float(np.std(accs)), 4),
               'cv5_balanced_accuracy': round(float(balanced_accuracy_score(all_yt, all_yp)), 4),
               'cv5_macro_f1': round(float(f1_score(all_yt, all_yp, average='macro')), 4)}
    with open(os.path.join(OUT, 'lstm_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: lstm_results.json, lstm_cm.png, lstm_curves.png")

if __name__ == "__main__": main()
