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

    def fit_biases(self, x):
        with torch.no_grad():
            self.conv.bias.data.zero_()
            out = self.conv(x)
            out = out.transpose(0, 1).reshape(self.num_kernels, -1)
            mins = out.min(dim=1)[0]
            maxs = out.max(dim=1)[0]
            rand_vals = torch.rand(self.num_kernels, device=out.device)
            new_biases = mins + rand_vals * (maxs - mins)
            self.conv.bias.data = -new_biases

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
                 head_epochs=30, head_lr=5e-4, batch_size=256, in_channels=20, seq_len=656):
        # Ensure we strictly use GPU if requested
        self.device = get_device()
        try:
            from engines_safe_print import safe_print
        except Exception:
            try:
                from src.engines_safe_print import safe_print
            except Exception:
                safe_print = print
        try:
            safe_print(f"[MiniRocketPipeline] Using device: {self.device}")
        except OSError:
            pass
        
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

    def fit(self, X, y, X_val=None, y_val=None, progress_callback=None):
        """
        X shape: (n_instances, n_channels, n_timepoints)
        y shape: (n_instances,)
        """
        start_time = time.time()

        if X.ndim == 2:
            raise ValueError("X must be 3D (batch, channels, time) for pure GPU MiniRocket.")

        self.classes_ = np.unique(y)
        n_cls = len(self.classes_)
        
        # 0) Fit Biases from a subset of data
        try:
            print("[MiniRocketPipeline] Fitting biases from training data...")
        except OSError:
            pass
        self.extractor.eval()
        subset_idx = np.random.choice(len(X), min(len(X), 128), replace=False)
        X_sub = torch.tensor(X[subset_idx], dtype=torch.float32).to(self.device)
        self.extractor.fit_biases(X_sub)
        
        # 1) Extract PPV features in batches using GPU
        try:
            print("[MiniRocketPipeline] Extracting PPV features on GPU...")
        except OSError:
            pass
        
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
        
        # 3) Train GPU MLP Head
        try:
            print("[MiniRocketPipeline] Training GPU MLP Head...")
        except OSError:
            pass
        lut = {c: i for i, c in enumerate(self.classes_)}
        yi = torch.tensor([lut[v] for v in y], dtype=torch.long)
        
        self.gpu_head = GPUMiniRocketHead(self.num_kernels, n_cls, self.hidden, self.dropout).to(self.device)
        
        class NormalizingDataset(torch.utils.data.Dataset):
            def __init__(self, X, y, mean, std):
                self.X = X
                self.y = y
                self.mean = mean.squeeze(0)
                self.std = std.squeeze(0)
            def __len__(self):
                return len(self.X)
            def __getitem__(self, idx):
                # Normalize on the fly to save RAM
                return (self.X[idx] - self.mean) / self.std, self.y[idx]
                
        ds = NormalizingDataset(X_transformed, yi, self.feat_mean, self.feat_std)
        dl = torch.utils.data.DataLoader(ds, batch_size=min(self.batch_size, len(ds)), shuffle=True)
        
        opt = torch.optim.AdamW(self.gpu_head.parameters(), lr=self.head_lr, weight_decay=5e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.head_epochs), eta_min=1e-6)
        
        # Add class weighting to fix mode collapse (all classes predicting a single class)
        from sklearn.utils.class_weight import compute_class_weight
        class_w = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
        class_w_tensor = torch.tensor(class_w, dtype=torch.float32).to(self.device)
        
        crit = nn.CrossEntropyLoss(weight=class_w_tensor, label_smoothing=0.01)
        
        amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if amp else None
        
        self.gpu_head.train()
        for epoch in range(self.head_epochs):
            total_loss = 0.0
            correct = 0
            total = 0
            
            self.gpu_head.train()
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
                preds = torch.argmax(logits, dim=1)
                correct += (preds == by).sum().item()
                total += by.size(0)
            sched.step()
            
            train_acc = correct / total if total > 0 else 0.0
            train_loss_avg = total_loss / len(dl) if len(dl) > 0 else 0.0
            
            val_loss_avg = 0.0
            val_acc = 0.0
            if X_val is not None and y_val is not None:
                self.gpu_head.eval()
                with torch.no_grad():
                    # Predict validation
                    # Note: for large validation sets we should batch, but for EEG it might fit
                    # Let's use predict_proba which handles batching internally
                    y_val_pred_probs = self.predict_proba(X_val)
                    y_val_pred = np.argmax(y_val_pred_probs, axis=1)
                    val_acc = float((y_val_pred == y_val).mean())
                    # Approx val loss (cross entropy):
                    # probabilities for true classes
                    true_probs = y_val_pred_probs[np.arange(len(y_val)), y_val]
                    # clip to avoid log(0)
                    true_probs = np.clip(true_probs, 1e-7, 1.0)
                    val_loss_avg = float(-np.mean(np.log(true_probs)))
            
            if progress_callback:
                progress_callback(epoch + 1, train_loss_avg, train_acc, val_loss_avg, val_acc)
            
        self.training_time = time.time() - start_time
        try:
            print(f"[MiniRocketPipeline] Done in {self.training_time:.2f}s")
        except OSError:
            pass
        return self

    def predict(self, X):
        idx = np.argmax(self.predict_proba(X), axis=1)
        if hasattr(self, 'classes_') and self.classes_ is not None:
            return self.classes_[idx]
        return idx

    def predict_proba(self, X):
        self.extractor.eval()
        self.gpu_head.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        
        # Safely handle legacy numpy arrays or PyTorch tensors
        m = self.feat_mean
        s = self.feat_std
        if not hasattr(m, 'to'):
            m = torch.tensor(m, dtype=torch.float32)
        if not hasattr(s, 'to'):
            s = torch.tensor(s, dtype=torch.float32)
            
        mu = m.to(self.device)
        sd = s.to(self.device)
        
        probs = []
        ppv_batch = 64  # reduced batch on GPU to avoid OOM
        with torch.no_grad():
            for i in range(0, len(X_t), ppv_batch):
                bx = X_t[i:i+ppv_batch].to(self.device)
                
                # Extract
                out = self.extractor(bx)
                
                # Normalize
                zn = (out - mu) / sd
                
                # Predict
                logits = self.gpu_head(zn)
                p = torch.softmax(logits, dim=1).cpu().numpy()
                probs.append(p)
                
        return np.vstack(probs)
    
    def save(self, filepath):
        if not hasattr(self, 'channel_names') or self.channel_names is None:
            raise ValueError("channel_names is mandatory for saving the model.")
        if not hasattr(self, 'sfreq') or self.sfreq is None:
            raise ValueError("sfreq is mandatory for saving the model.")
            
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
            'seq_len': self.seq_len,
            'channel_names': self.channel_names,
            'sfreq': self.sfreq,
            'label_classes': getattr(self, 'classes_', None),
        }
        torch.save(state, filepath)

    def load(self, filepath):
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        self.num_kernels = state.get('num_kernels', 10000)
        self.hidden = state.get('hidden', 512)
        self.dropout = state.get('dropout', 0.15)
        self.in_channels = state.get('in_channels', 20)
        self.seq_len = state.get('seq_len', 656)
        
        self.channel_names = state.get('channel_names')
        if self.channel_names is None:
            # Legacy fallback
            self.channel_names = [f"EEG_{i}" for i in range(self.in_channels)]
            
        self.sfreq = state.get('sfreq')
        if self.sfreq is None:
            # Legacy fallback
            self.sfreq = 160.0
        
        self.extractor = TorchMiniRocket(self.in_channels, self.seq_len, self.num_kernels).to(self.device)
        self.extractor.load_state_dict(state['extractor'])
        
        n_cls = len(state['classes_'])
        self.gpu_head = GPUMiniRocketHead(self.num_kernels, n_cls, self.hidden, self.dropout).to(self.device)
        self.gpu_head.load_state_dict(state['gpu_head'])
        
        self.feat_mean = state['feat_mean']
        self.feat_std = state['feat_std']
        self.classes_ = state['classes_']
        self.training_time = state.get('training_time', 0.0)
