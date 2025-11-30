import numpy as np
import pandas as pd
import os
import joblib
from scipy.signal import welch
from sklearn.metrics import accuracy_score, confusion_matrix

# ==========================================
# 1. SETUP
# ==========================================
DATA_DIR = "simulated_sessions"
MODEL_FILENAME = "meditation_svm_hard.pkl"
SCALER_FILENAME = "scaler_hard.pkl"
FS = 256
# GAMMA EXCLUDED
BANDS = {'Delta': 2, 'Theta': 6, 'Alpha': 10, 'Beta': 20}
CATEGORIES = {0: 'Relaxed', 1: 'Focus', 2: 'Deep', 3: 'Distraction'}
ID_TO_LABEL = CATEGORIES

# ==========================================
# 2. FEATURE EXTRACTION
# ==========================================
def get_features_from_csv(filepath):
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' not found.")
        return None
    
    try:
        df = pd.read_csv(filepath)
        raw = df.values
        
        # Validation checks
        if raw.shape[0] < FS:
            print(f"Error: Data too short ({raw.shape[0]} samples). Needs at least {FS} samples.")
            return None
        if raw.shape[1] < 4:
            print(f"Error: Data has less than 4 channels ({raw.shape[1]} columns).")
            return None

        features = []
        for ch in range(4):
            sig = raw[:, ch]
            # Use appropriate nperseg
            nperseg = min(512, len(sig))
            freqs, psd = welch(sig, FS, nperseg=nperseg)
            
            for band in BANDS.keys():
                center = BANDS[band]
                idx = np.where((freqs >= center-2) & (freqs <= center+2))
                features.append(np.sum(psd[idx]))
        return np.array(features)
        
    except Exception as e:
        print(f"Error processing CSV: {e}")
        return None

# ==========================================
# 3. TESTING LOGIC
# ==========================================
def run_test_single_file(filename="testing_hard_mode.csv"):
    if not os.path.exists(MODEL_FILENAME):
        print("Model not found. Run training_hard_mode.py first.")
        return

    file_path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(file_path):
        # Fallback: Check if file exists in current directory if not in DATA_DIR
        if os.path.exists(filename):
            file_path = filename
        else:
            print(f"Error: Test file '{filename}' not found in '{DATA_DIR}/' or current directory.")
            return

    print("--- Loading AI Brain ---")
    try:
        svm = joblib.load(MODEL_FILENAME)
        scaler = joblib.load(SCALER_FILENAME)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print(f"\n--- Testing on file: {filename} ---")
    
    # 1. Extract Features
    feat = get_features_from_csv(file_path)
    if feat is None: return

    # 2. Scale
    try:
        feat_scaled = scaler.transform([feat])
    except ValueError as e:
        print(f"Scaling Error: {e}. Feature count mismatch.")
        return

    # 3. Predict
    pred_id = svm.predict(feat_scaled)[0]
    pred_name = CATEGORIES[pred_id]
    
    # 4. Confidence
    probs = svm.predict_proba(feat_scaled)[0]
    conf = probs[pred_id] * 100
    
    print("-" * 40)
    print(f"AI Prediction: {pred_name}")
    print(f"Confidence:    {conf:.1f}%")
    print("-" * 40)
    print("Probability Breakdown:")
    for idx, prob in enumerate(probs):
        print(f"  {CATEGORIES[idx]:<12}: {prob*100:.1f}%")
    print("-" * 40)

if __name__ == "__main__":
    # You can change this to test different files
    target_file = "testing_hard_mode.csv"
    run_test_single_file(target_file)