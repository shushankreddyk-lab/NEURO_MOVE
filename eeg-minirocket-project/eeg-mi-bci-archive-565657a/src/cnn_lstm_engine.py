import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np

try:
    torch.serialization.add_safe_globals([np._core.multiarray._reconstruct])
except Exception:
    pass


def get_device(prefer="auto"):
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SqueezeExcite1D(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        hid = max(1, channels // reduction)
        self.fc = nn.Sequential(nn.Linear(channels, hid), nn.SiLU(),
                                nn.Linear(hid, channels), nn.Sigmoid())

    def forward(self, x):
        b, c, _ = x.shape
        w = self.fc(self.pool(x).view(b, c)).view(b, c, 1)
        return x * w

class MultiScaleConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, p_drop=0.1, drop_path=0.0):
        super().__init__()
        b = out_ch // 4
        self.b1 = nn.Sequential(nn.Conv1d(in_ch, b, 3, padding=1), nn.BatchNorm1d(b), nn.SiLU())
        self.b2 = nn.Sequential(nn.Conv1d(in_ch, b, 7, padding=3), nn.BatchNorm1d(b), nn.SiLU())
        self.b3 = nn.Sequential(nn.Conv1d(in_ch, b, 7, padding=9, dilation=3), nn.BatchNorm1d(b), nn.SiLU())
        self.b4 = nn.Sequential(nn.Conv1d(in_ch, b, 7, padding=18, dilation=6), nn.BatchNorm1d(b), nn.SiLU())
        self.proj = nn.Sequential(nn.Conv1d(out_ch, out_ch, 1), nn.BatchNorm1d(out_ch))
        self.se = SqueezeExcite1D(out_ch)
        self.drop = nn.Dropout(p_drop)
        self.short = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()
        self.drop_path = drop_path
    def forward(self, x):
        y = torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)
        y = self.drop(self.se(self.proj(y)))
        s = self.short(x)
        if self.training and self.drop_path > 0 and torch.rand(1).item() < self.drop_path:
            return self.act(s)
        return self.act(y + s)


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


class CNN_LSTM_Network(nn.Module):
    def __init__(self, num_classes=4, channels=20, samples=656,
                 cnn_width=64, lstm_hidden=64, tf_layers=2, tf_heads=4, p_drop=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = channels
        
        # Simplified lightweight CNN stem for easy overfitting
        self.stem = nn.Sequential(
            nn.Conv1d(channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, cnn_width, kernel_size=5, padding=2),
            nn.BatchNorm1d(cnn_width),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        
        self.lstm = nn.LSTM(input_size=cnn_width, hidden_size=lstm_hidden,
                            num_layers=1, batch_first=True, bidirectional=True, dropout=0.0)
                            
        d = lstm_hidden * 2
        self.dropout = nn.Dropout(p_drop)
        self.fc1 = nn.Linear(d, 64)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)
        
        # Keep aliases for older save compatibility (save/load mapping)
        self.conv1 = self.stem[0]

    def forward(self, x):
        if x.dim() == 4:
            x = x.squeeze(1)
        x = self.stem(x)
        x = x.permute(0, 2, 1)  # (B, T, C)
        x, _ = self.lstm(x)     # (B, T, D)
        # Global average pooling over time
        x = x.mean(dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc3(x)

class CNN_LSTM_Pipeline:
    def __init__(self, epochs=100, batch_size=64, lr=3e-3, channels=10, samples=656, num_classes=4,
                 device="auto", amp=True, patience=50,
                 cnn_width=64, lstm_hidden=64, tf_layers=2, tf_heads=4,
                 dropout=0.25, label_smoothing=0.05, task_type="classification"):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = get_device(device)
        self.amp = amp and self.device.type == "cuda"
        self.patience = patience
        self.scaler = torch.amp.GradScaler("cuda") if self.amp else None
        self.model = CNN_LSTM_Network(
            num_classes=num_classes, channels=channels, cnn_width=cnn_width,
            lstm_hidden=lstm_hidden, tf_layers=tf_layers, tf_heads=tf_heads,
            p_drop=0.1).to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.0)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, epochs))
        # NOTE: raw LR 3e-3 is for normalized data; effective LR is warmed up in fit()
        self.training_time = 0.0
        self.mean = None
        self.std = None
        self.history = {"loss": [], "val_loss": [], "acc": [], "val_acc": [], "lr": []}
        self.channels = channels
        self.task_type = task_type
        if self.task_type == "regression":
            self.criterion = nn.MSELoss()

    def _prepare_data(self, X):
        # Expects (Batch, Channels, Samples) like (B, 74, 656)
        # Needs (Batch, 1, Channels, Samples) for Conv1D compatibility block, 
        # but our new CNN_LSTM_Network takes (Batch, Channels, Samples) or (Batch, 1, Channels, Samples) and squeezes it.
        if X.ndim == 3:
            X = X[:, np.newaxis, :, :]
        elif X.ndim == 2:
            # Fallback if flat
            X = X[:, np.newaxis, 10, 656] # This would fail, but just a guard
        return X

    def fit(self, X, y, X_val=None, y_val=None, progress_callback=None, incremental=False):
        X = self._prepare_data(X)
            
        n_out = self.model.fc3.out_features
        y = np.asarray(y)
        
        if self.task_type == "classification":
            # Handle string/object labels
            if y.dtype.kind in {'U', 'S', 'O'} or (y_val is not None and y_val.dtype.kind in {'U', 'S', 'O'}):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y = le.fit_transform(y)
                if y_val is not None:
                    y_val = le.transform(y_val)
                self.label_classes_ = le.classes_
            else:
                # Remap arbitrary labels (e.g. MNE event ids) to 0..K-1 for CrossEntropyLoss
                uniq = np.unique(y)
                if uniq.min() < 0 or uniq.max() >= n_out or len(uniq) != n_out:
                    self.label_classes_ = uniq
                    lut = {c: i for i, c in enumerate(uniq)}
                    y = np.array([lut[v] for v in y], dtype=np.int64)
                    if y_val is not None:
                        new_y_val = []
                        for v in np.asarray(y_val):
                            if v not in lut:
                                raise ValueError(f"Unknown validation label: {v}. Must be one of {list(lut.keys())}")
                            new_y_val.append(lut[v])
                        y_val = np.array(new_y_val, dtype=np.int64)
                    n_out = len(uniq)
                else:
                    self.label_classes_ = np.arange(n_out)
            # Use minlength to ensure all classes are represented in weights
            class_counts = np.bincount(y, minlength=n_out)
            total_samples = len(y)
            class_weights = np.ones(len(class_counts), dtype=np.float32)
            for c in range(len(class_counts)):
                if class_counts[c] > 0:
                    class_weights[c] = total_samples / (len(class_counts) * class_counts[c])
            self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(self.device), label_smoothing=0.0)
        else:
            # Regression task
            self.label_classes_ = np.arange(n_out) # Dummy for dashboard compatibility
            self.criterion = nn.MSELoss()
            
        # Z-score normalization per channel: mean/std over (batch, time), keep (1, C, 1, 1)
        batch_mean = np.mean(X, axis=(0, 3), keepdims=True)
        batch_std = np.std(X, axis=(0, 3), keepdims=True) + 1e-8

        if incremental and self.mean is not None:
            alpha = 0.1  # EMA weight for new data
            m = np.asarray(self.mean, dtype=np.float64).reshape(1, -1, 1, 1)
            s = np.asarray(self.std, dtype=np.float64).reshape(1, -1, 1, 1)
            self.mean = (1 - alpha) * m + alpha * batch_mean
            self.std = (1 - alpha) * s + alpha * batch_std
        else:
            self.mean = batch_mean
            self.std = batch_std
        # Guard dead channels (zero variance -> scale 1.0, never divide by ~0)
        std = np.asarray(self.std, dtype=np.float64)
        std[std < 1e-6] = 1.0
        self.std = std

        X = (X - self.mean) / self.std
            
        y_dtype = torch.float32 if self.task_type == "regression" else torch.long
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32), 
            torch.tensor(y, dtype=y_dtype)
        )
        # Reverted back to num_workers=0 (default) because PyTorch multiprocessing on Windows CPU causes severe overhead
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        if X_val is not None:
            X_val = self._prepare_data(X_val)
            X_val = (X_val - self.mean) / self.std
            val_X_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            val_y_t = torch.tensor(y_val, dtype=y_dtype).to(self.device)

        start_time = time.time()
        
        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad()
                if self.amp:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(batch_X)
                        loss = self.criterion(outputs, batch_y)
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(batch_X)
                    loss = self.criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                running_loss += loss.item() * batch_X.size(0)
                if self.task_type == "classification":
                    _, predicted = torch.max(outputs.data, 1)
                    correct += (predicted == batch_y).sum().item()
                else:
                    # R^2 score equivalent metric (or just negative loss)
                    # For simplicity, we just use -loss as accuracy so higher is better for early stopping
                    correct += -loss.item() * batch_X.size(0)
                    
                total += batch_y.size(0)

            train_loss = running_loss / total
            train_acc = correct / total if self.task_type == "classification" else -train_loss
            self.scheduler.step()

            val_loss, val_acc = 0.0, 0.0
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(val_X_t)
                    v_loss = self.criterion(val_outputs, val_y_t)
                    val_loss = v_loss.item()
                    if self.task_type == "classification":
                        _, v_pred = torch.max(val_outputs.data, 1)
                        val_acc = (v_pred == val_y_t).sum().item() / val_y_t.size(0)
                    else:
                        val_acc = -val_loss
            self.history["loss"].append(train_loss)
            self.history["acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            try:
                self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
            except Exception:
                pass
            # early stopping on val accuracy
            if X_val is not None:
                if not hasattr(self, 'best_val_acc'):
                    self.best_val_acc = -1.0
                    self.best_state_dict = None
                    self.patience_counter = 0
                
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.best_state_dict = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                
                if self.patience_counter >= self.patience:
                    if progress_callback:
                        progress_callback(epoch + 1, train_loss, train_acc, val_loss, val_acc)
                    break

            if progress_callback:
                progress_callback(epoch + 1, train_loss, train_acc, val_loss, val_acc)

        if hasattr(self, 'best_state_dict') and self.best_state_dict is not None:
            self.model.load_state_dict(self.best_state_dict)

        self.training_time = time.time() - start_time
        return self

    def predict(self, X):
        self.model.eval()
        X = self._prepare_data(X)
        if self.mean is not None:
            X = (X - self.mean) / self.std
        with torch.no_grad():
            batch_X = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(batch_X)
            if self.task_type == "classification":
                _, predicted = torch.max(outputs.data, 1)
                idx = predicted.cpu().numpy()
                if hasattr(self, 'label_classes_') and self.label_classes_ is not None:
                    return self.label_classes_[idx]
                return idx
            else:
                return outputs.cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        X = self._prepare_data(X)
        if self.mean is not None:
            X = (X - self.mean) / self.std
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(X_t)
            if self.task_type == "classification":
                probs = torch.nn.functional.softmax(outputs, dim=1)
            else:
                probs = outputs # For regression, return raw outputs instead of softmax
        return probs.cpu().numpy()
        
    def measure_latency(self, X_sample):
        self.model.eval()
        # Expect single sample: (74, 656) -> (1, 1, 74, 656)
        if X_sample.ndim == 2:
            X_sample = X_sample[np.newaxis, np.newaxis, :, :]
        elif X_sample.ndim == 3:
            X_sample = X_sample[np.newaxis, :, :, :]
            
        if self.mean is not None:
            X_sample = (X_sample - self.mean) / self.std
            
        X_t = torch.tensor(X_sample, dtype=torch.float32).to(self.device)
        start = time.perf_counter()
        with torch.no_grad():
            _ = self.model(X_t)
        end = time.perf_counter()
        return (end - start) * 1000.0

    def save(self, filepath):
        state = {
            'model_state_dict': self.model.state_dict(),
            'mean': self.mean,
            'std': self.std,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'lr': self.lr,
            'label_classes': getattr(self, 'label_classes_', None),
            'channels': self.model.conv1.in_channels,
            'channel_names': getattr(self, 'channel_names', None),
            'sfreq': getattr(self, 'sfreq', 160.0),
            'num_classes': self.model.fc3.out_features,
            'task_type': getattr(self, 'task_type', 'classification'),
            'model_cfg': {
                'cnn_width': self.model.lstm.input_size,
                'lstm_hidden': self.model.lstm.hidden_size,
            },
        }
        torch.save(state, filepath)

    def load(self, filepath):
        state = torch.load(filepath, map_location=self.device, weights_only=False)

        saved_arch = state.get('arch', 'legacy')
        saved_cfg = state.get('model_cfg', {}) or {}
        # Infer architecture parameters to avoid size mismatch
        if 'channels' in state and 'num_classes' in state:
            channels = state['channels']
            num_classes = state['num_classes']
        else:
            # Fallback inference from state_dict shape for older models
            sd = state['model_state_dict']
            if 'conv1.weight' in sd and sd['conv1.weight'].dim() == 3:
                channels = sd['conv1.weight'].shape[1]
                num_classes = sd['fc3.weight'].shape[0]
            else:
                channels = sd['depthwise_conv.weight'].shape[2]
                num_classes = sd['fc.weight'].shape[0]

        # Dynamically reconstruct model with correct shape + saved hyper-params
        self.model = CNN_LSTM_Network(
            num_classes=num_classes, channels=channels,
            cnn_width=int(saved_cfg.get('cnn_width', 64)),
            lstm_hidden=int(saved_cfg.get('lstm_hidden', 64)),
        ).to(self.device)

        try:
            self.model.load_state_dict(state['model_state_dict'])
        except Exception as e:
            print("Warning: Could not load state_dict, likely due to architecture change. Please retrain.", e)
        
        self.mean = state.get('mean', None)
        self.std = state.get('std', None)
        self.label_classes_ = state.get('label_classes', None)
        self.task_type = state.get('task_type', 'classification')
        if self.task_type == "regression":
            self.criterion = nn.MSELoss()
        self.epochs = state.get('epochs', 10)
        self.batch_size = state.get('batch_size', 64)
        self.lr = state.get('lr', 1e-3)
        return self
