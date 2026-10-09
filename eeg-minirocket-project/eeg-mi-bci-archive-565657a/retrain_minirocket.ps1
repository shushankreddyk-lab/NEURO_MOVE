$ErrorActionPreference = "Stop"

Write-Host "Retraining MiniRocket on PhysioNet..."
python src/train_master.py --mode master --dataset "D:\eeg-minirocket-project\dataset\eeg-motor-movementimagery-dataset-1.0.0" --model MiniRocket --epochs 30 --sub_start 1 --sub_end 10

Write-Host "Retraining MiniRocket on BCI 2a..."
python src/train_master.py --mode master --dataset "D:\eeg-minirocket-project\dataset\BCICIV_2a_mat" --model MiniRocket --epochs 30 --sub_start 1 --sub_end 10

Write-Host "Retraining MiniRocket on WAY-EEG-GAL..."
python src/train_master.py --mode master --dataset "way" --model MiniRocket --epochs 30 --sub_start 1 --sub_end 10

python src/train_master.py --mode master --dataset "" --model MiniRocket --epochs 30 --sub_start 1 --sub_end 10

Write-Host "Retraining MiniRocket on High-Gamma..."
python src/train_master.py --mode master --dataset "high-gamma" --model MiniRocket --epochs 30 --sub_start 1 --sub_end 10

Write-Host "Retraining MiniRocket on Kaya..."
python src/train_master.py --mode master --dataset "kaya" --model MiniRocket --epochs 30 --sub_start 1 --sub_end 10

Write-Host "Done!"
