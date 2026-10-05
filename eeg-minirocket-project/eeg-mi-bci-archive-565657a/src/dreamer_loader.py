import numpy as np
import scipy.io
import sys

class DREAMERLoader:
    def __init__(self, mat_path, window_size_sec=1.0, overlap=0.5, sub_start=1, sub_end=10):
        self.mat_path = mat_path
        self.window_size_sec = window_size_sec
        self.overlap = overlap
        self.sub_start = sub_start
        self.sub_end = sub_end

    def load_data(self):
        print(f"Loading DREAMER from {self.mat_path}...")
        mat = scipy.io.loadmat(self.mat_path)
        dreamer = mat['DREAMER'][0, 0]
        data = dreamer['Data'][0]
        
        sfreq = int(dreamer['EEG_SamplingRate'][0, 0])
        window_samples = int(self.window_size_sec * sfreq)
        step = int(window_samples * (1.0 - self.overlap))
        
        X_list = []
        y_list = []
        
        # Adjust subject indices (1-based to 0-based)
        start_idx = max(0, self.sub_start - 1)
        end_idx = min(len(data), self.sub_end)
        
        for subj_idx in range(start_idx, end_idx):
            subj = data[subj_idx]
            eeg = subj['EEG'][0, 0]
            stimuli = eeg['stimuli'][0, 0] # Shape (18, 1)
            
            # Using Valence for classes
            valence = subj['ScoreValence'][0, 0] # shape (18, 1)? Wait, in previous log it was (18,) or (1, 18). Let's handle both.
            
            for video_idx in range(len(stimuli)):
                trial_data = stimuli[video_idx][0]  # Shape: (timepoints, 14)
                
                # valence might be a 1x18 array or similar
                try:
                    val_score = valence[video_idx][0] if isinstance(valence[video_idx], np.ndarray) else valence[video_idx]
                except Exception:
                    val_score = valence[0][video_idx]
                    
                # Classify: 0 if Valence <= 2, 1 if Valence == 3, 2 if Valence >= 4
                if val_score <= 2:
                    label = 0
                elif val_score == 3:
                    label = 1
                else:
                    label = 2
                    
                for start in range(0, len(trial_data) - window_samples, step):
                    window = trial_data[start : start + window_samples, :]
                    X_list.append(window.T)  # Transpose to (channels, timepoints)
                    y_list.append(label)
                    
        X_all = np.array(X_list, dtype=np.float32)
        y_all = np.array(y_list, dtype=int)
        
        print(f"DREAMER loaded -> X shape: {X_all.shape}, y shape: {y_all.shape}")
        return X_all, y_all

if __name__ == "__main__":
    loader = DREAMERLoader(r"D:\eeg-minirocket-project\dataset\DREAMER.mat", sub_start=1, sub_end=2)
    X, y = loader.load_data()
