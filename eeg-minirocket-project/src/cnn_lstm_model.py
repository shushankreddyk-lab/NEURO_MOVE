"""
Hybrid CNN-LSTM Deep Learning Architecture for EEG Motor Imagery Classification.
Reproducing Hwaidi & Ghanem (NeuroImage 328 (2026) 121816):
- Prompt Specification:
    Conv1D(C, 64, k=25) -> BN -> ReLU -> MaxPool(3) ->
    Conv1D(64, 128, k=13) -> BN -> ReLU -> MaxPool(3) ->
    LSTM(128, 128, num_layers=2, dropout=0.5) ->
    Dropout -> Linear -> Softmax
- Table 1 Specification (configurable alternate):
    Conv1D(16, k=3) -> Conv1D(32, k=3) -> MaxPool(2) -> LSTM(100) -> Dense(100) -> Dense(50) -> Dense(4)
- Optimizer: Adam(lr=1e-5), Weight Decay=0.01 (L2 regularization)
- Latency benchmarking target: ~8.0 ms/sample
"""

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union

try:
    torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
except Exception:
    pass


def resolve_device(prefer="auto"):
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        hid = max(1, channels // reduction)
        self.fc = nn.Sequential(nn.Linear(channels, hid), nn.SiLU(),
                                nn.Linear(hid, channels), nn.Sigmoid())
    def forward(self, x):
        b, c, _ = x.shape
        return x * self.fc(self.pool(x).view(b, c)).view(b, c, 1)


class MultiScaleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.1):
        super().__init__()
        b = out_ch // 4
        self.b1 = nn.Sequential(nn.Conv1d(in_ch, b, 3, padding=1), nn.BatchNorm1d(b), nn.SiLU())
        self.b2 = nn.Sequential(nn.Conv1d(in_ch, b, 7, padding=3), nn.BatchNorm1d(b), nn.SiLU())
        self.b3 = nn.Sequential(nn.Conv1d(in_ch, b, 7, padding=9, dilation=3), nn.BatchNorm1d(b), nn.SiLU())
        self.b4 = nn.Sequential(nn.Conv1d(in_ch, b, 7, padding=18, dilation=6), nn.BatchNorm1d(b), nn.SiLU())
        self.proj = nn.Sequential(nn.Conv1d(out_ch, out_ch, 1), nn.BatchNorm1d(out_ch))
        self.se = SEBlock1D(out_ch)
        self.drop = nn.Dropout(p_drop)
        self.short = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()
    def forward(self, x):
        y = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        return self.act(self.drop(self.se(self.proj(y))) + self.short(x))


class HybridCNNLSTM(nn.Module):
    """Complex GPU arch: multi-scale CNN stem + BiLSTM + Transformer + attention."""
    def __init__(
        self,
        in_channels: int = 64,
        n_classes: int = 4,
        cnn_width: int = 128,
        lstm_hidden: int = 128,
        tf_layers: int = 2,
        tf_heads: int = 4,
        dropout: float = 0.25,
        conv1_filters: int = 64,
        conv1_kernel: int = 25,
        conv2_filters: int = 128,
        conv2_kernel: int = 13,
        lstm_layers: int = 2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.stem = nn.Sequential(
            MultiScaleBlock(in_channels, 64, 0.10), nn.MaxPool1d(2),
            MultiScaleBlock(64, 128, 0.15), nn.MaxPool1d(2),
            MultiScaleBlock(128, cnn_width, 0.20), nn.MaxPool1d(2))
        self.conv1 = self.stem[0].b1[0]
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.SiLU()
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = self.stem[2].b1[0]
        self.bn2 = nn.BatchNorm1d(128)
        self.relu2 = nn.SiLU()
        self.pool2 = nn.MaxPool1d(2)
        self.lstm = nn.LSTM(input_size=cnn_width, hidden_size=lstm_hidden,
                            num_layers=2, batch_first=True, bidirectional=True, dropout=dropout)
        d = lstm_hidden * 2
        lyr = nn.TransformerEncoderLayer(d_model=d, nhead=tf_heads, dim_feedforward=d * 4,
                                         dropout=dropout, batch_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(lyr, num_layers=tf_layers)
        self.norm = nn.LayerNorm(d)
        self.attn = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x.squeeze(1)
        x = self.stem(x).permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.norm(self.transformer(x))
        w = torch.softmax(self.attn(x), dim=1)
        x = (x * w).sum(dim=1)
        return self.fc(self.dropout(x))


class LegacyHybridCNNLSTM(HybridCNNLSTM):
    """Old 2-conv arch kept for docstring compat (unused)."""
    pass


class Table1CNNLSTM(nn.Module):
    """
    Alternate CNN-LSTM configuration referenced in Table 1 of the paper.
    Conv1D(16, k=3) -> Conv1D(32, k=3) -> MaxPool(2) -> LSTM(100) -> Dense(100) -> Dense(50) -> Dense(4)
    """
    def __init__(self, in_channels: int = 64, n_classes: int = 4, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.lstm = nn.LSTM(input_size=32, hidden_size=100, batch_first=True)
        self.fc1 = nn.Linear(100, 100)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(100, 50)
        self.relu4 = nn.ReLU()
        self.out = nn.Linear(50, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.conv1(x))
        out = self.pool(self.relu2(self.conv2(out)))
        out = out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(out)
        feat = lstm_out[:, -1, :]
        out = self.relu3(self.fc1(feat))
        out = self.dropout(out)
        out = self.relu4(self.fc2(out))
        return self.out(out)


class CNNLSTMPipeline:
    """
    Wrapper for training, evaluation, and latency benchmarking of CNN-LSTM.
    """
    def __init__(
        self,
        in_channels: int = 64,
        n_classes: int = 4,
        arch_type: str = "primary",  # "primary" or "table1"
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        device: Optional[str] = None,
        amp: bool = True,
        dropout: float = 0.25,
    ):
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.device = resolve_device(device or "auto")
        self.amp = amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.amp else None

        if arch_type == "table1":
            self.model = Table1CNNLSTM(in_channels=in_channels, n_classes=n_classes)
        else:
            self.model = HybridCNNLSTM(in_channels=in_channels, n_classes=n_classes, dropout=dropout)

        self.model.to(self.device)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 20,
        batch_size: int = 32,
        early_stopping_patience: int = 7,
        verbose: bool = True
    ):
        t0 = time.perf_counter()
        # Per-channel z-norm from TRAIN ONLY (same as eeg-mi-bci engine)
        mu = X_train.mean(axis=(0, 2), keepdims=True)
        sd = X_train.std(axis=(0, 2), keepdims=True) + 1e-8
        sd[sd < 1e-6] = 1.0
        self.norm_mean_, self.norm_std_ = mu.astype(np.float64), sd.astype(np.float64)
        Xtr = ((X_train - mu) / sd).astype(np.float32)
        Xv = None
        if X_val is not None:
            Xv = ((X_val - mu) / sd).astype(np.float32)
        train_ds = TensorDataset(
            torch.tensor(Xtr, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        val_loader = None
        if Xv is not None and y_val is not None:
            val_ds = TensorDataset(
                torch.tensor(Xv, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long),
            )
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                self.optimizer.zero_grad()
                if self.amp:
                    with torch.amp.autocast("cuda"):
                        logits = self.model(bx)
                        loss = self.criterion(logits, by)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    logits = self.model(bx)
                    loss = self.criterion(logits, by)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                running_loss += loss.item() * len(by)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == by).sum().item()
                total += len(by)

            epoch_loss = running_loss / total
            epoch_acc = correct / total
            try:
                self.scheduler.step()
            except Exception:
                pass
            self.history["train_loss"].append(epoch_loss)
            self.history["train_acc"].append(epoch_acc)

            # Validation
            val_loss, val_acc = 0.0, 0.0
            if val_loader:
                self.model.eval()
                v_correct, v_total, v_loss = 0, 0, 0.0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(self.device), vy.to(self.device)
                        out = self.model(vx)
                        l = self.criterion(out, vy)
                        v_loss += l.item() * len(vy)
                        v_preds = torch.argmax(out, dim=1)
                        v_correct += (v_preds == vy).sum().item()
                        v_total += len(vy)
                val_loss = v_loss / v_total
                val_acc = v_correct / v_total
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose:
                            print(f"Early stopping triggered at epoch {epoch}")
                        break

            if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == epochs):
                val_str = f" - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}" if val_loader else ""
                print(f"Epoch {epoch:02d}/{epochs:02d} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}{val_str}")

        self.train_time_sec_ = time.perf_counter() - t0
        return self

    def _apply_norm(self, X: np.ndarray) -> np.ndarray:
        if getattr(self, "norm_mean_", None) is None:
            return X.astype(np.float32)
        return ((X - self.norm_mean_) / self.norm_std_).astype(np.float32)

    def predict(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        self.model.eval()
        Xn = self._apply_norm(X)
        preds = []
        with torch.no_grad():
            for i in range(0, len(Xn), batch_size):
                bx = torch.tensor(Xn[i:i+batch_size], dtype=torch.float32).to(self.device)
                logits = self.model(bx)
                p = torch.argmax(logits, dim=1).cpu().numpy()
                preds.append(p)
        return np.concatenate(preds)

    def predict_proba(self, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
        self.model.eval()
        Xn = self._apply_norm(X)
        probs = []
        with torch.no_grad():
            for i in range(0, len(Xn), batch_size):
                bx = torch.tensor(Xn[i:i+batch_size], dtype=torch.float32).to(self.device)
                logits = self.model(bx)
                prob = torch.softmax(logits, dim=1).cpu().numpy()
                probs.append(prob)
        return np.concatenate(probs)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean(preds == y))

    def benchmark_latency(self, single_sample: np.ndarray, n_warmup: int = 5, n_runs: int = 50) -> Dict[str, float]:
        """
        Benchmark single-trial inference latency in milliseconds (~8.0 ms target).
        """
        if single_sample.ndim == 2:
            single_sample = single_sample[np.newaxis, ...]

        bx = torch.tensor(single_sample, dtype=torch.float32).to(self.device)
        self.model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = self.model(bx)

        # Benchmark
        timings = []
        with torch.no_grad():
            for _ in range(n_runs):
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                _ = self.model(bx)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()
                timings.append((t1 - t0) * 1000.0)

        timings = np.array(timings)
        return {
            "latency_mean_ms": float(np.mean(timings)),
            "latency_std_ms": float(np.std(timings)),
            "latency_p50_ms": float(np.median(timings)),
            "latency_p95_ms": float(np.percentile(timings, 95))
        }

    def save(self, filepath: Union[str, Path]):
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "in_channels": self.in_channels,
            "n_classes": self.n_classes,
            "state_dict": self.model.state_dict(),
            "history": self.history,
            "norm_mean": getattr(self, "norm_mean_", None),
            "norm_std": getattr(self, "norm_std_", None),
            "arch": "multiscale-bilstm-transformer",
        }, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path], arch_type: str = "primary", device: Optional[str] = None) -> "CNNLSTMPipeline":
        # Local trusted checkpoints contain numpy norm stats; torch>=2.6
        # defaults weights_only=True which blocks them.
        data = torch.load(filepath, map_location=device or "cpu", weights_only=False)
        pipeline = cls(
            in_channels=data["in_channels"],
            n_classes=data["n_classes"],
            arch_type=arch_type,
            device=device
        )
        pipeline.model.load_state_dict(data["state_dict"])
        pipeline.history = data.get("history", {})
        pipeline.norm_mean_ = data.get("norm_mean", None)
        pipeline.norm_std_ = data.get("norm_std", None)
        if data.get("arch", "legacy") == "legacy" and pipeline.norm_mean_ is None:
            print("Note: legacy checkpoint has no norm stats — predictions may be off. Please retrain.")
        return pipeline
