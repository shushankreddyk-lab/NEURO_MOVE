import sys

# --- Force CUDA torch from D:\pip_packages (overrides system CPU torch) ---
_D_PKGS = r"D:\pip_packages"
if _D_PKGS not in sys.path:
    sys.path.insert(0, _D_PKGS)

import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import json

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class EEG_Conformer(nn.Module):
    """
    Advanced EEG-Conformer Architecture:
    1. Spatial-Temporal Convolutional Module
    2. Multi-Head Self-Attention Module
    3. Fully Connected Classification Module
    """
    def __init__(self, num_classes=4, channels=20, samples=656, 
                 F1=64, kernLength=64, pool1=8, F2=64, depthMultiplier=2,
                 drop_prob=0.3, d_model=80, heads=8, tf_layers=4):
        super().__init__()
        self.in_channels = channels
        self.samples = samples
        self.num_classes = num_classes
        
        # 1. Temporal Convolution (1, channels, samples) -> (F1, channels, samples)
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding="same", bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1)
        
        # 2. Spatial Convolution (F1, channels, samples) -> (F1*D, 1, samples)
        self.depthwise = nn.Conv2d(F1, F1 * depthMultiplier, (channels, 1), 
                                   groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * depthMultiplier)
        self.elu = nn.ELU()
        
        # 3. Pooling and Dropout
        self.avg_pool = nn.AvgPool2d((1, pool1))
        self.dropout = nn.Dropout(drop_prob)
        
        # 4. Projection to Transformer d_model
        # output of pool: (F1*D, 1, samples//pool1)
        # We need it in shape: (batch, seq_len, d_model)
        out_len = samples // pool1
        
        # If F1*D != d_model, we project it
        self.proj = nn.Conv1d(F1 * depthMultiplier, d_model, 1)
        
        # 5. Self-Attention (Transformer)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=d_model*4, 
            dropout=drop_prob, batch_first=True, activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=tf_layers)
        
        # 6. Classification Head
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * out_len, 512),
            nn.BatchNorm1d(512),
            nn.ELU(),
            nn.Dropout(drop_prob),
            nn.Linear(512, 128),
            nn.ELU(),
            nn.Dropout(drop_prob * 0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, channels, samples)
        # Add channel dim for Conv2D: (batch, 1, channels, samples)
        x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.batchnorm1(x)
        
        x = self.depthwise(x)
        x = self.batchnorm2(x)
        x = self.elu(x)
        
        x = self.avg_pool(x)
        x = self.dropout(x)
        
        # Squeeze the empty spatial dimension: (batch, F1*D, samples//pool)
        x = x.squeeze(2)
        
        # Project to d_model
        x = self.proj(x)
        
        # Permute for Transformer: (batch, seq_len, d_model)
        x = x.permute(0, 2, 1)
        
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        logits = self.fc(x)
        return logits


class AdvancedEEGPipeline:
    """
    End-to-End Deep Learning pipeline wrapping the EEG-Conformer.
    Executes entirely on GPU.
    """
    def __init__(self, num_classes=4, channels=20, samples=656, 
                 epochs=15, batch_size=256, lr=1e-3, device="cuda", task_type="classification"):
        self.device = get_device()
        try:
            print(f"[AdvancedEEGPipeline] Using device: {self.device}")
        except OSError:
            pass
        
        self.num_classes = num_classes
        self.channels = channels
        self.samples = samples
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.task_type = task_type
        
        self.model = EEG_Conformer(
            num_classes=num_classes, 
            channels=channels, 
            samples=samples
        ).to(self.device)
        
        self.classes_ = None
        self.training_time = 0.0
        
        # Standardization parameters
        self.feat_mean = None
        self.feat_std = None

    def fit(self, X, y, X_val=None, y_val=None):
        start_time = time.time()
        
        self.classes_ = np.unique(y) if self.task_type == "classification" else np.arange(y.shape[1] if y.ndim > 1 else 1)
        
        # Calculate standardization on raw signals
        X_t = torch.tensor(X, dtype=torch.float32)
        self.feat_mean = X_t.mean(dim=0, keepdim=True)
        self.feat_std = X_t.std(dim=0, keepdim=True) + 1e-8
        
        X_t = (X_t - self.feat_mean) / self.feat_std
        
        y_dtype = torch.float32 if self.task_type == "regression" else torch.long
        
        if self.task_type == "classification":
            lut = {c: i for i, c in enumerate(self.classes_)}
            yi = torch.tensor([lut[v] for v in y], dtype=y_dtype)
        else:
            yi = torch.tensor(y, dtype=y_dtype)
        
        ds = torch.utils.data.TensorDataset(X_t, yi)
        # Using pin_memory for faster transfer
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        
        if X_val is not None and y_val is not None:
            X_v = torch.tensor(X_val, dtype=torch.float32)
            X_v = (X_v - self.feat_mean) / self.feat_std
            if self.task_type == "classification":
                try:
                    y_v = torch.tensor([lut[v] for v in y_val], dtype=y_dtype)
                except KeyError as e:
                    raise ValueError(f"Unknown validation label: {e.args[0]}. Must be one of {list(lut.keys())}")
            else:
                y_v = torch.tensor(y_val, dtype=y_dtype)
            val_ds = torch.utils.data.TensorDataset(X_v, y_v)
            val_dl = torch.utils.data.DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)
        else:
            val_dl = None
            
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=self.lr, steps_per_epoch=len(dl), epochs=self.epochs
        )
        
        # Loss
        if self.task_type == "classification":
            class_counts = np.bincount(yi.numpy())
            total = len(yi)
            class_weights = total / (len(self.classes_) * class_counts)
            class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)
            crit = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
        else:
            crit = nn.MSELoss()
        
        amp = self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if amp else None
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            train_correct = 0
            train_total = 0
            for bx, by in dl:
                bx, by = bx.to(self.device), by.to(self.device)
                opt.zero_grad()
                if amp:
                    with torch.amp.autocast("cuda"):
                        logits = self.model(bx)
                        loss = crit(logits, by)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    logits = self.model(bx)
                    loss = crit(logits, by)
                    loss.backward()
                    opt.step()
                sched.step()
                total_loss += loss.item()
                
                if self.task_type == "classification":
                    preds = logits.detach().argmax(dim=1)
                    train_correct += (preds == by).sum().item()
                else:
                    train_correct += -loss.item() * by.size(0)
                train_total += by.size(0)
                
            train_acc = train_correct / train_total if self.task_type == "classification" else train_correct / train_total
                
            if val_dl:
                self.model.eval()
                correct = 0
                total_val = 0
                val_loss_sum = 0.0
                with torch.no_grad():
                    for bx, by in val_dl:
                        bx, by = bx.to(self.device), by.to(self.device)
                        logits = self.model(bx)
                        loss = crit(logits, by)
                        val_loss_sum += loss.item() * bx.size(0)
                        if self.task_type == "classification":
                            preds = logits.argmax(dim=1)
                            correct += (preds == by).sum().item()
                        else:
                            correct += -loss.item() * bx.size(0)
                        total_val += by.size(0)
                val_acc = correct / total_val if self.task_type == "classification" else correct / total_val
                avg_val_loss = val_loss_sum / total_val if total_val > 0 else 0.0
                avg_train_loss = total_loss / len(dl)
                print(json.dumps({
                    "type": "epoch", 
                    "epoch": epoch + 1, 
                    "total_epochs": self.epochs,
                    "train_loss": avg_train_loss, 
                    "val_loss": avg_val_loss,
                    "train_acc": train_acc, 
                    "val_acc": val_acc
                }), flush=True)
            
        self.training_time = time.time() - start_time
        try:
            print(f"[AdvancedEEGPipeline] Done in {self.training_time:.2f}s")
        except OSError:
            pass
        return self

    def predict(self, X):
        if self.task_type == "classification":
            idx = np.argmax(self.predict_proba(X), axis=1)
            if hasattr(self, 'classes_') and self.classes_ is not None:
                return self.classes_[idx]
            return idx
        else:
            return self.predict_proba(X)

    def predict_proba(self, X):
        self.model.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        
        # Safely handle legacy numpy arrays or PyTorch tensors
        m = self.feat_mean
        s = self.feat_std
        if hasattr(m, 'cpu'):
            m = m.cpu()
        else:
            m = torch.tensor(m, dtype=torch.float32)
            
        if hasattr(s, 'cpu'):
            s = s.cpu()
        else:
            s = torch.tensor(s, dtype=torch.float32)
            
        X_t = (X_t - m) / s
        
        probs = []
        with torch.no_grad():
            for i in range(0, len(X_t), self.batch_size):
                bx = X_t[i:i+self.batch_size].to(self.device)
                logits = self.model(bx)
                if self.task_type == "classification":
                    p = torch.softmax(logits, dim=1).cpu().numpy()
                else:
                    p = logits.cpu().numpy()
                probs.append(p)
                
        return np.vstack(probs)
    
    def save(self, filepath):
        state = {
            'model': self.model.state_dict(),
            'feat_mean': self.feat_mean,
            'feat_std': self.feat_std,
            'classes_': self.classes_,
            'training_time': self.training_time,
            'num_classes': self.num_classes,
            'channels': self.channels,
            'channel_names': getattr(self, 'channel_names', None),
            'sfreq': getattr(self, 'sfreq', 160.0),
            'label_classes': getattr(self, 'label_classes_', None),
            'samples': self.samples,
            'task_type': getattr(self, 'task_type', 'classification')
        }
        torch.save(state, filepath)

    def load(self, filepath):
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        self.num_classes = state['num_classes']
        self.channels = state['channels']
        self.samples = state['samples']
        self.task_type = state.get('task_type', 'classification')
        self.feat_mean = state.get('feat_mean', None)
        self.feat_std = state.get('feat_std', None)
        
        self.model = EEG_Conformer(self.num_classes, self.channels, self.samples).to(self.device)
        try:
            self.model.load_state_dict(state['model'])
        except RuntimeError as e:
            if "size mismatch" in str(e) or "Missing key" in str(e) or "Unexpected key" in str(e):
                try:
                    print("[AdvancedEEGPipeline] Modern model load failed. Falling back to legacy architecture...", flush=True)
                except OSError:
                    pass
                # Initialize legacy architecture
                self.model = EEG_Conformer(
                    self.num_classes, self.channels, self.samples,
                    F1=40, kernLength=64, pool1=8, F2=40, depthMultiplier=2,
                    drop_prob=0.5, d_model=40, heads=4, tf_layers=3
                )
                # Revert to legacy FC
                out_len = self.samples // 8
                self.model.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(40 * out_len, 256),
                    nn.ELU(),
                    nn.Dropout(0.5),
                    nn.Linear(256, self.num_classes)
                )
                self.model = self.model.to(self.device)
                self.model.load_state_dict(state['model'], strict=False)
                try:
                    print("[AdvancedEEGPipeline] Legacy architecture loaded successfully.", flush=True)
                except OSError:
                    pass
            else:
                raise e
        
        self.feat_mean = state['feat_mean']
        self.feat_std = state['feat_std']
        self.classes_ = state['classes_']
        self.training_time = state.get('training_time', 0.0)
        
        self.model.eval()
