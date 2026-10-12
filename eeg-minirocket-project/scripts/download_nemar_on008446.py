import os
import subprocess
import sys
import time

def install_openneuro():
    print("Installing openneuro-py...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openneuro-py"])

def download_dataset():
    dataset_id = "ds008446"
    target_dir = r"D:\eeg-minirocket-project\datasets\nemar_on008446"
    
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
        
    print(f"Downloading {dataset_id} to {target_dir}...")
    import openneuro
    
    max_retries = 50
    for attempt in range(max_retries):
        try:
            print(f"--- Download Attempt {attempt + 1} ---")
            openneuro.download(dataset=dataset_id, target_dir=target_dir)
            print("Download entirely complete!")
            break
        except Exception as e:
            print(f"Network error encountered: {e}")
            print("Retrying in 10 seconds... (already downloaded files will be skipped)")
            time.sleep(10)

if __name__ == "__main__":
    try:
        import openneuro
    except ImportError:
        install_openneuro()
    
    download_dataset()
