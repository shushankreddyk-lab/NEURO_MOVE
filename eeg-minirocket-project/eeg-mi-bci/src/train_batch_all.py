import os
import sys
import subprocess
import time
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--kernels", type=int, default=10000)
    parser.add_argument("--partition", type=int, default=80)
    args = parser.parse_args()

    total_subjects = 109
    batch_size = 20
    
    # Path to train_master.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_script = os.path.join(script_dir, 'train_master.py')
    
    print(json.dumps({"type": "info", "message": f"Starting full dataset training ({total_subjects} subjects in batches of {batch_size})..."}))
    
    start_time = time.time()
    
    for start_idx in range(1, total_subjects + 1, batch_size):
        end_idx = min(start_idx + batch_size - 1, total_subjects)
        
        batch_msg = f"Training Batch: Subjects {start_idx} to {end_idx}"
        print(json.dumps({"type": "batch_start", "message": batch_msg, "sub_start": start_idx, "sub_end": end_idx}))
        print(json.dumps({"type": "reset_chart"})) # Tell UI to reset the chart for the new batch
        
        cmd = [
            sys.executable,
            train_script,
            "--mode", "master",
            "--dataset", args.dataset,
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--kernels", str(args.kernels),
            "--partition", str(args.partition),
            "--sub_start", str(start_idx),
            "--sub_end", str(end_idx)
        ]
        
        try:
            # Run the training script for this batch
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Stream output
            for line in process.stdout:
                line = line.strip()
                if line:
                    try:
                        # Try to parse as JSON to forward it
                        parsed = json.loads(line)
                        print(json.dumps(parsed), flush=True)
                    except json.JSONDecodeError:
                        # If not JSON, wrap it
                        print(json.dumps({"type": "info", "message": line}), flush=True)
                        
            process.wait()
            
            if process.returncode != 0:
                err_msg = process.stderr.read()
                print(json.dumps({"type": "error", "message": f"Batch {start_idx}-{end_idx} failed: {err_msg}"}), flush=True)
            else:
                print(json.dumps({"type": "batch_success", "message": f"Batch {start_idx}-{end_idx} completed successfully!"}), flush=True)
                
        except Exception as e:
            print(json.dumps({"type": "error", "message": f"Exception in batch {start_idx}-{end_idx}: {str(e)}"}), flush=True)
            
    total_time = time.time() - start_time
    print(json.dumps({"type": "complete", "message": f"Full dataset training finished in {total_time:.2f} seconds!"}), flush=True)

if __name__ == "__main__":
    main()
