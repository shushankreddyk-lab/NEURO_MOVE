$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$base = "D:\eeg-minirocket-project"
$python = "$base\.venv\Scripts\python.exe"
$script = "$base\eeg-mi-bci\src\train_master.py"
$dataset = "D:\eeg-minirocket-project\physionet"
$models = @(
    @{ model = "CSP+LDA"; epochs = 1 },
    @{ model = "MiniRocket"; epochs = 150 },
    @{ model = "CNN-LSTM"; epochs = 50 },
    @{ model = "EEGNet"; epochs = 80 },
    @{ model = "Shallow ConvNet"; epochs = 100 },
    @{ model = "Deep ConvNet"; epochs = 100 }
)
foreach ($m in $models) {
    Write-Host "=== Starting $($m.model) ===" -ForegroundColor Cyan
    & $python $script --mode master --dataset $dataset --model $m.model --sub_start 1 --sub_end 76 --epochs $m.epochs
    Write-Host "=== $($m.model) DONE (exit $LASTEXITCODE) ===" -ForegroundColor Green
    Start-Sleep -Seconds 5
}
Write-Host "ALL DONE!" -ForegroundColor Yellow
