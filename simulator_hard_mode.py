import numpy as np
import pandas as pd
import os
import shutil

# ==========================================
# 1. CONFIGURATION
# ==========================================
DATA_DIR = "simulated_sessions"
if os.path.exists(DATA_DIR): shutil.rmtree(DATA_DIR)
os.makedirs(DATA_DIR)

FS = 256
CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']
DURATION_SEC = 60

# Brainwave Bands
BANDS = {'Delta': 2, 'Theta': 6, 'Alpha': 10, 'Beta': 20, 'Gamma': 40}

# "Recipes" for each state (Amplitudes)
STATE_RECIPES = {
    'Relaxed':     {'Delta': 0.2, 'Theta': 0.2, 'Alpha': 1.2, 'Beta': 0.2, 'Gamma': 0.1},
    'Focus':       {'Delta': 0.1, 'Theta': 0.9, 'Alpha': 0.4, 'Beta': 0.3, 'Gamma': 0.2},
    'Deep':        {'Delta': 0.9, 'Theta': 0.6, 'Alpha': 0.1, 'Beta': 0.1, 'Gamma': 0.0},
    'Distraction': {'Delta': 0.1, 'Theta': 0.1, 'Alpha': 0.2, 'Beta': 0.9, 'Gamma': 0.4}
}

# Base Ratings for each state (Target values on 1-5 scale)
# We will add random noise to these so not every "Relaxed" person gives a 5/5
RATING_PROFILES = {
    'Relaxed':     {'Depth': 3, 'Vitakka': 3, 'Piti': 3, 'Sukha': 4, 'Ekaggata': 3},
    'Focus':       {'Depth': 4, 'Vitakka': 2, 'Piti': 4, 'Sukha': 3, 'Ekaggata': 5}, # High Focus
    'Deep':        {'Depth': 5, 'Vitakka': 1, 'Piti': 5, 'Sukha': 5, 'Ekaggata': 5}, # High Everything
    'Distraction': {'Depth': 1, 'Vitakka': 5, 'Piti': 1, 'Sukha': 2, 'Ekaggata': 1}  # Low Focus, High Thought
}

# ==========================================
# 2. GENERATION LOGIC
# ==========================================
def generate_signal(duration, state_name):
    t = np.linspace(0, duration, int(duration * FS), endpoint=False)
    signal = np.zeros_like(t)
    recipe = STATE_RECIPES[state_name]
    
    for band, freq in BANDS.items():
        amp = recipe[band] * np.random.uniform(0.8, 1.2)
        jitter = np.sin(2 * np.pi * 0.1 * t)
        phase = 2 * np.pi * np.cumsum(freq + jitter) / FS
        signal += amp * np.sin(phase)
        
    noise = np.random.normal(0, 0.5, len(t))
    return signal + noise

def generate_varied_qa(state_name):
    """Generates realistic, varied survey answers based on the state."""
    profile = RATING_PROFILES[state_name]
    
    # Helper to add randomness but keep within 1-5 range
    def randomize(val):
        # Increased variety: range of -2 to +1
        # Example for target 5: can become 3, 4, 5 (weighted towards 5)
        # Example for target 1: can become 1, 2, 3 (weighted towards 1)
        
        # Probabilities: [ -2,  -1,   0,  +1 ]
        noise = np.random.choice([-2, -1, 0, 1], p=[0.1, 0.2, 0.5, 0.2])
        new_val = val + noise
        return max(1, min(5, new_val)) # Clamp between 1 and 5

    qa_data = {
        'Participant_ID': f'User_{np.random.randint(1000, 9999)}',
        'Date': '2025-10-15',
        'Session_Time': f'{np.random.randint(8, 22)}:00',
        
        # Part 1: Overall Depth
        'Q1_Depth': randomize(profile['Depth']),
        
        # Part 2: The 5 Factors
        'Q2_Vitakka': randomize(profile['Vitakka']),
        'Q3_Piti': randomize(profile['Piti']),
        'Q4_Sukha': randomize(profile['Sukha']),
        'Q5_Ekaggata': randomize(profile['Ekaggata']),
        
        # Part 3: Ground Truth (The Label)
        # We assume the user correctly identifies their state for training data
        'Label_Class': state_name 
    }
    return qa_data

def run_simulation(num_files_per_state=10):
    print(f"--- Generating Data Pairs (EEG + QA) in '{DATA_DIR}/' ---")
    
    for state in STATE_RECIPES.keys():
        for i in range(1, num_files_per_state + 1):
            # A. Generate EEG Data
            raw_data = np.array([generate_signal(DURATION_SEC, state) for _ in range(4)]).T
            eeg_filename = f"sim_{state}_{i}.csv"
            pd.DataFrame(raw_data, columns=CHANNELS).to_csv(os.path.join(DATA_DIR, eeg_filename), index=False)
            
            # B. Generate Varied Questionnaire Data
            qa_data = generate_varied_qa(state)
            qa_filename = f"qa_{state}_{i}.csv"
            pd.DataFrame([qa_data]).to_csv(os.path.join(DATA_DIR, qa_filename), index=False)
            
            # Print the varied ratings to confirm they are changing
            rating_summary = f"{qa_data['Q1_Depth']}{qa_data['Q2_Vitakka']}{qa_data['Q3_Piti']}{qa_data['Q4_Sukha']}{qa_data['Q5_Ekaggata']}"
            print(f"Created Pair: {eeg_filename} (Ratings: {rating_summary})")

if __name__ == "__main__":
    run_simulation(num_files_per_state=10)