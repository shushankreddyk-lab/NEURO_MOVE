import numpy as np
import scipy.io as sio
import os
import mne

class DREAMERLoader:
    def __init__(self, mat_path, window_size_sec=1.0, overlap=0.5):
        """
        DREAMER Dataset Loader
        Loads the .mat file containing EEG data (14 channels) and Valence/Arousal/Dominance scores.
        
        Args:
            mat_path (str): Path to DREAMER.mat
            window_size_sec (float): Size of each EEG window in seconds.
            overlap (float): Overlap fraction between windows (0 to 1).
        """
        self.mat_path = mat_path
        self.window_size_sec = window_size_sec
        self.overlap = overlap
        
        # Load MAT file
        print(f"Loading DREAMER dataset from {self.mat_path}...")
        self.mat = sio.loadmat(self.mat_path)
        self.dataset = self.mat['DREAMER'][0, 0]
        
        self.sfreq = int(self.dataset['EEG_SamplingRate'][0,0])
        
        # Extract channel names
        raw_ch_names = self.dataset['EEG_Electrodes'][0]
        self.ch_names = [ch[0][0] for ch in raw_ch_names]
        
        self.window_samples = int(self.window_size_sec * self.sfreq)
        self.step_size = int(self.window_samples * (1 - self.overlap))
        
    def load_data(self):
        """
        Extracts sliding windows from the 18 video stimuli for all 23 subjects.
        Returns:
            X (np.ndarray): Shape (n_samples, n_channels, n_timesteps)
            y (np.ndarray): Shape (n_samples, 3) -> [Valence, Arousal, Dominance]
        """
        X_all = []
        y_all = []
        
        subjects = self.dataset['Data'][0]
        num_subjects = len(subjects)
        print(f"Found {num_subjects} subjects.")
        
        for subj_idx in range(num_subjects):
            subj_data = subjects[subj_idx]
            
            # shape: (18, 1)
            stimuli = subj_data['EEG'][0,0]['stimuli'][0,0]
            valence = subj_data['ScoreValence'][0,0]
            arousal = subj_data['ScoreArousal'][0,0]
            dominance = subj_data['ScoreDominance'][0,0]
            
            for video_idx in range(18):
                # shape: (time_steps, 14 channels)
                eeg_trial = stimuli[video_idx, 0] 
                v_score = valence[video_idx, 0]
                a_score = arousal[video_idx, 0]
                d_score = dominance[video_idx, 0]
                
                # Sliding window over the trial
                n_timesteps = eeg_trial.shape[0]
                
                for start in range(0, n_timesteps - self.window_samples + 1, self.step_size):
                    end = start + self.window_samples
                    window = eeg_trial[start:end, :]  # (samples, 14)
                    
                    # Store as (channels, samples)
                    X_all.append(window.T)
                    y_all.append([v_score, a_score, d_score])
                    
        X = np.array(X_all, dtype=np.float32)
        y = np.array(y_all, dtype=np.float32)
        
        print(f"Loaded DREAMER Data: X={X.shape}, y={y.shape} (Valence, Arousal, Dominance)")
        return X, y

if __name__ == "__main__":
    loader = DREAMERLoader(r"d:\eeg-minirocket-project\dataset\DREAMER.mat", window_size_sec=1.0)
    X, y = loader.load_data()
