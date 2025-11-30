import numpy as np
import pandas as pd
import os
import joblib
from scipy.signal import welch
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# ==========================================
# 1. SETUP
# ==========================================
DATA_DIR = "simulated_sessions"
MODEL_FILENAME = "meditation_svm_hard.pkl"
SCALER_FILENAME = "scaler_hard.pkl"
FS = 256
# GAMMA EXCLUDED
BANDS = {'Delta': 2, 'Theta': 6, 'Alpha': 10, 'Beta': 20}
CATEGORIES = {'Relaxed': 0, 'Focus': 1, 'Deep': 2, 'Distraction': 3}
ID_TO_LABEL = {v: k for k, v in CATEGORIES.items()}

# ==========================================
# 2. FEATURE EXTRACTION
# ==========================================
def get_features_from_csv(filepath):
    if not os.path.exists(filepath): return None
    df = pd.read_csv(filepath)
    raw = df.values
    features = []
    for ch in range(4):
        sig = raw[:, ch]
        freqs, psd = welch(sig, FS, nperseg=FS*2)
        # Only iterate through bands defined in BANDS (No Gamma)
        for band in BANDS.keys():
            center = BANDS[band]
            idx = np.where((freqs >= center-2) & (freqs <= center+2))
            features.append(np.sum(psd[idx]))
    return np.array(features)

def get_label_from_qa(qa_filepath):
    if not os.path.exists(qa_filepath): return None
    df = pd.read_csv(qa_filepath)
    state_name = df['Label_Class'].iloc[0] 
    return CATEGORIES.get(state_name)

# ==========================================
# 3. TRAINING LOGIC
# ==========================================
class BrainWaveTrainer:
    def __init__(self):
        self.svm = SVC(kernel='rbf', probability=True)
        self.scaler = StandardScaler()
        self.X_train = []
        self.y_train = []

    def load_data(self):
        print("--- Loading Training Data (Files 1-8 Only) ---")
        all_files = os.listdir(DATA_DIR)
        sim_files = [f for f in all_files if f.startswith('sim_') and f.endswith('.csv')]
        
        # STRICT FILTER: Only use files with index 1 to 8
        train_files = []
        for f in sim_files:
            try:
                # Extract the index number from filename like 'sim_Relaxed_5.csv'
                file_index = int(f.split('_')[-1].split('.')[0])
                if file_index <= 8:
                    train_files.append(f)
            except ValueError:
                continue # Skip files that don't match pattern
        
        print(f"Found {len(train_files)} training files.")

        for eeg_file in train_files:
            parts = eeg_file.split('_')
            # robustly construct QA filename
            if len(parts) >= 3:
                qa_file = f"qa_{parts[1]}_{parts[2]}"
            else:
                continue

            eeg_path = os.path.join(DATA_DIR, eeg_file)
            qa_path = os.path.join(DATA_DIR, qa_file)
            
            feat = get_features_from_csv(eeg_path)
            label_id = get_label_from_qa(qa_path)
            
            if feat is not None and label_id is not None:
                self.X_train.append(feat)
                self.y_train.append(label_id)
                # print(f"Learned: {eeg_file} -> {ID_TO_LABEL[label_id]}") # Commented out to reduce noise

    def train_and_save(self):
        print("\n--- Training Model ---")
        X = np.array(self.X_train)
        y = np.array(self.y_train)
        
        if len(X) == 0:
            print("Error: No training data loaded!")
            return

        X_scaled = self.scaler.fit_transform(X)
        self.svm.fit(X_scaled, y)
        
        joblib.dump(self.svm, MODEL_FILENAME)
        joblib.dump(self.scaler, SCALER_FILENAME)
        print(f"Model saved to '{MODEL_FILENAME}'")

    def practice_test(self):
        """Tests the model on files 9 and 10 to give a final score."""
        print("\n--- Running Final Practice Test (On Unseen Files 9 & 10) ---")
        
        all_files = os.listdir(DATA_DIR)
        sim_files = [f for f in all_files if f.startswith('sim_') and f.endswith('.csv')]

        # STRICT FILTER: Only use files with index > 8
        test_files = []
        for f in sim_files:
            try:
                file_index = int(f.split('_')[-1].split('.')[0])
                if file_index > 8:
                    test_files.append(f)
            except ValueError:
                continue
        
        results = []
        y_true = []
        y_pred = []
        
        print(f"{'File':<25} | {'Actual (Truth)':<15} | {'Testing (Predicted)':<20} | {'Status'}")
        print("-" * 75)

        for eeg_file in test_files:
            parts = eeg_file.split('_')
            qa_file = f"qa_{parts[1]}_{parts[2]}"
            
            eeg_path = os.path.join(DATA_DIR, eeg_file)
            qa_path = os.path.join(DATA_DIR, qa_file)
            
            # 1. Get Truth
            true_id = get_label_from_qa(qa_path)
            if true_id is None: continue
            true_name = ID_TO_LABEL[true_id]
            
            # 2. Predict
            feat = get_features_from_csv(eeg_path)
            if feat is None: continue

            # Use the scaler we just trained
            feat_scaled = self.scaler.transform([feat])
            pred_id = self.svm.predict(feat_scaled)[0]
            pred_name = ID_TO_LABEL[pred_id]
            
            status = "✅" if true_id == pred_id else "❌"
            
            print(f"{eeg_file:<25} | {true_name:<15} | {pred_name:<20} | {status}")
            
            y_true.append(true_id)
            y_pred.append(pred_id)
            
        if len(y_true) > 0:
            acc = accuracy_score(y_true, y_pred)
            print("-" * 75)
            print(f"Final Test Score: {acc*100:.1f}%")
        else:
            print("No test files found.")

if __name__ == "__main__":
    trainer = BrainWaveTrainer()
    trainer.load_data()
    trainer.train_and_save()
    trainer.practice_test() # Run validation immediately