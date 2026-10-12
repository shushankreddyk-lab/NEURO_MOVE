import os
import mne
import moabb
from moabb.datasets import (
    Schirrmeister2017,  # High-Gamma (Right hand, Left hand, Rest, Feet / Grasps)
    BNCI2014_001,       # 4-Class BCI Comp IV 2a (Left Hand, Right Hand, Feet, Tongue)
    PhysionetMI,        # PhysioNet 109 subjects (Left/Right Fist, Both Fists [Up], Both Feet [Down])
    Cho2017,            # Left vs Right Hand MI (64 channels)
)

DATA_DIR = r"D:\eeg-minirocket-project\data"
os.makedirs(DATA_DIR, exist_ok=True)
mne.set_config('MNE_DATASETS_MOABB_PATH', DATA_DIR)

def download_datasets():
    print(f"=== Downloading Verified EEG Datasets to {DATA_DIR} ===\n")
    
    # 1. BCI Competition IV 2a (Tongue, Left Hand, Right Hand, Feet)
    try:
        print("1. [Tongue & 4-Class MI] Downloading BNCI2014_001 (BCI Comp IV 2a)...")
        dataset_bnci = BNCI2014_001()
        dataset_bnci.get_data(subjects=[1, 2])
        print(" -> BNCI2014_001 (Tongue MI) downloaded successfully!\n")
    except Exception as e:
        print(f" -> Error downloading BNCI2014_001: {e}\n")

    # 2. PhysioNet MI (Left/Right Hand, Both Fists [Up], Both Feet [Down])
    try:
        print("2. [Directional / Fists / Feet] Downloading PhysioNet MI...")
        dataset_physio = PhysionetMI()
        dataset_physio.get_data(subjects=[1, 2])
        print(" -> PhysioNet MI downloaded successfully!\n")
    except Exception as e:
        print(f" -> Error downloading PhysioNet MI: {e}\n")

    # 3. High-Gamma Dataset (Schirrmeister 2017)
    try:
        print("3. [High-Gamma] Downloading Schirrmeister2017 (G-Node GIN)...")
        dataset_hgd = Schirrmeister2017()
        dataset_hgd.get_data(subjects=[1])
        print(" -> High-Gamma Dataset downloaded successfully!\n")
    except Exception as e:
        print(f" -> Error downloading High-Gamma: {e}")
        print("    Direct G-Node Download: https://web.gin.g-node.org/robintibor/high-gamma-dataset/\n")

if __name__ == "__main__":
    download_datasets()

