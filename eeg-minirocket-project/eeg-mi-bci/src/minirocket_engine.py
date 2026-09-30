import numpy as np
import time
import sys

# --- Force CUDA torch from D:\pip_packages (overrides system CPU torch) ---
_D_PKGS = r"D:\pip_packages"
if _D_PKGS not in sys.path:
    sys.path.insert(0, _D_PKGS)

import torch
import torch.nn as nn
from scipy.special import softmax

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TorchMiniRocket(nn.Module):
    """
    Pure PyTorch implementation of a MiniRocket-like feature extractor.
    Executes entirely on GPU (or CPU) without sktime dependency.
    """
    def __init__(self, in_channels, seq_len, num_kernels=10000):
        super().__init__()
        self.num_kernels = num_kernels
        self.seq_len = seq_len
        self.in_channels = in_channels
        
        # We simulate 10,000 kernels of length 9
        kernel_size = 9
        
        # Create Conv1d
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=num_kernels,
            kernel_size=kernel_size,
            padding="same",
            bias=True
        )
        
        # Freeze weights and biases
        self.conv.weight.requires_grad = False
        self.conv.bias.requires_grad = False
        
        # Initialize with MiniRocket-like weights (e.g. mostly -1, some 2)
        # For simplicity in pure PyTorch, we randomly initialize from Normal dist
        # and then threshold/scale them.
        nn.init.normal_(self.conv.weight, mean=0.0, std=1.0)
        # Force weights to be either -1 or 2 to mimic MiniRocket's discrete weights
        mask = (self.conv.weight > 0.0)
        self.conv.weight.data[mask] = 2.0
        self.conv.weight.data[~mask] = -1.0
        
        # Initialize biases to cover a broad range of quantiles
        nn.init.uniform_(self.conv.bias, a=-5.0, b=5.0)

    def forward(self, x):
        # x: (batch, in_channels, seq_len)
        out = self.conv(x)  # (batch, num_kernels, seq_len)
        # Proportion of Positive Values (PPV)
        ppv = (out > 0).float().mean(dim=-1)  # (batch, num_kernels)
        return ppv


class GPUMiniRocketHead(nn.Module):
    """GPU classifier head over frozen MiniRocket features."""
    def __init__(self, in_dim, n_classes, hidden=512, p_drop=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden, hidden // 2), nn.LayerNorm(hidden // 2), nn.GELU(), nn.Dropout(p_drop),
            nn.Linear(hidden // 2, n_classes)
        )
    def forward(self, x):
        return self.net(x)


class MiniRocketPipeline:
    def __init__(self, num_kernels=10000, device="cuda", hidden=512, dropout=0.15,
                 head_epochs=150, head_lr=5e-4, batch_size=256, in_channels=20, seq_len=656):
        # Ensure we strictly use GPU if requested
        self.device = get_device()
        print(f"[MiniRocketPipeline] Using device: {self.device}")
        
        self.num_kernels = num_kernels
        self.hidden = hidden
        self.dropout = dropout
        self.head_epochs = head_epochs
        self.head_lr = head_lr
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.seq_len = seq_len
        
        self.extractor = TorchMiniRocket(in_channels, seq_len, num_kernels).to(self.device)
        self.extractor.eval() # Freeze extractor
        
        self.gpu_head = None
        self.feat_mean = None
        self.feat_std = None
        self.classes_ = None
        self.training_time = 0.0

    def fit(self, X, y):
        """
        X shape: (n_instances, n_channels, n_timepoints)
        y shape: (n_instances,)
        """
        start_time = time.time()

        if X.ndim == 2:
            raise ValueError("X must be 3D (batch, channels, time) for pure GPU MiniRocket.")

        self.classes_ = np.unique(y)
        n_cls = len(self.classes_)
        
        # 1) Extract PPV features in batches using GPU
        print("[MiniRocketPipeline] Extracting PPV features on GPU...")
        self.extractor.eval()
        
        X_t = torch.tensor(X, dtype=torch.float32)
        ppv_features = []
        ppv_batch = 64  # reduced batch on GPU to avoid OOM
        with torch.no_grad():
            for i in range(0, len(X_t), ppv_batch):
                batch_x = X_t[i:i+ppv_batch].to(self.device)
                out = self.extractor(batch_x)
                ppv_features.append(out.cpu())
                
        X_transformed = torch.cat(ppv_features, dim=0)
        
        # 2) Standardize features
        self.feat_mean = X_transformed.mean(dim=0, keepdim=True)
        self.feat_std = X_transformed.std(dim=0, keepdim=True) + 1e-8
        
        Zn = (X_transformed - self.feat_mean) / self.feat_std
        
        # 3) Train GPU MLP Head
        print("[MiniRocketPipeline] Training GPU MLP Head...")
        lut = {c: i for i, c in enumerate(self.classes_)}
        yi = torch.tensor([lut[v] for v in y], dtype=torch.long)
        
        self.gpu_head = GPUMiniRocketHead(self.num_kernels, n_cls, self.hidden, self.dropout).to(self.device)
        
        ds = torch.utils.data.TensorDataset(Zn, yi)
        dl = torch.utils.data.DataLoader(ds, batch_size=min(self.batch_size, len(ds)), shuffle=True)
        
        opt = torch.optim.AdamW(self.gpu_head.parameters(), lr=self.head_lr, weight_decay=5e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.head_epochs), eta_min=1e-6)
        crit = nn.CrossEntropyLoss(label_smoothing=0.01)
        
        amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if amp else None
        
        self.gpu_head.train()
        for epoch in range(self.head_epochs):
            total_loss = 0.0
            for bx, by in dl:
                bx, by = bx.to(self.device), by.to(self.device)
                opt.zero_grad()
                if amp:
                    with torch.amp.autocast("cuda"):
                        logits = self.gpu_head(bx)
                        loss = crit(logits, by)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    logits = self.gpu_head(bx)
                    loss = crit(logits, by)
                    loss.backward()
                    opt.step()
                total_loss += loss.item()
            sched.step()
            
        self.training_time = time.time() - start_time
        print(f"[MiniRocketPipeline] Done in {self.training_time:.2f}s")
        return self

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        self.extractor.eval()
        self.gpu_head.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        
        probs = []
        ppv_batch = 64  # reduced batch on GPU to avoid OOM
        with torch.no_grad():
            for i in range(0, len(X_t), ppv_batch):
                bx = X_t[i:i+ppv_batch].to(self.device)
                
                # Extract
                out = self.extractor(bx)
                
                # Normalize
                mu = self.feat_mean.to(self.device)
                sd = self.feat_std.to(self.device)
                zn = (out - mu) / sd
                
                # Predict
                logits = self.gpu_head(zn)
                p = torch.softmax(logits, dim=1).cpu().numpy()
                probs.append(p)
                
        return np.vstack(probs)
    
    def save(self, filepath):
        state = {
            'extractor': self.extractor.state_dict(),
            'gpu_head': self.gpu_head.state_dict(),
            'feat_mean': self.feat_mean,
            'feat_std': self.feat_std,
            'classes_': self.classes_,
            'training_time': self.training_time,
            'num_kernels': self.num_kernels,
            'hidden': self.hidden,
            'dropout': self.dropout,
            'in_channels': self.in_channels,
            'seq_len': self.seq_len
        }
        torch.save(state, filepath)

    def load(self, filepath):
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        self.num_kernels = state['num_kernels']
        self.hidden = state['hidden']
        self.dropout = state['dropout']
        self.in_channels = state['in_channels']
        self.seq_len = state['seq_len']
        
        self.extractor = TorchMiniRocket(self.in_channels, self.seq_len, self.num_kernels).to(self.device)
        self.extractor.load_state_dict(state['extractor'])
        
        n_cls = len(state['classes_'])
        self.gpu_head = GPUMiniRocketHead(self.num_kernels, n_cls, self.hidden, self.dropout).to(self.device)
        self.gpu_head.load_state_dict(state['gpu_head'])
        
        self.feat_mean = state['feat_mean']
        self.feat_std = state['feat_std']
        self.classes_ = state['classes_']
        self.training_time = state.get('training_time', 0.0)
