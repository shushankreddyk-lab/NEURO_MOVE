import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
import json

def get_device(prefer="auto"):
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class EEGNet(nn.Module):
    def __init__(self, num_classes=4, channels=22, samples=656, 
                 F1=32, D=2, F2=64, kernel_length=64, p_drop=0.25):
        super(EEGNet, self).__init__()
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.num_classes = num_classes
        self.channels = channels

        # Block 1
        self.conv1 = nn.Conv2d(1, self.F1, (1, kernel_length), padding='same', bias=False)
        self.batchnorm1 = nn.BatchNorm2d(self.F1)
        
        # Depthwise Conv
        self.depthwise = nn.Conv2d(self.F1, self.F1 * self.D, (channels, 1), groups=self.F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(self.F1 * self.D)
        self.elu1 = nn.GELU()
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(p_drop)

        # Block 2
        # Separable Conv = Depthwise + Pointwise
        self.sep_conv_depth = nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, 16), 
                                        padding='same', groups=self.F1 * self.D, bias=False)
        self.sep_conv_point = nn.Conv2d(self.F1 * self.D, self.F2, (1, 1), bias=False)
        self.batchnorm3 = nn.BatchNorm2d(self.F2)
        self.elu2 = nn.GELU()
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(p_drop)

        # Self-Attention (Transformer) Block
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.F2, nhead=4, dim_feedforward=self.F2*2, batch_first=True, dropout=p_drop)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Classifier
        # Output shape after pool2: (F2, 1, samples // 32)
        out_samples = samples // 32
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.F2 * out_samples, num_classes)

    def forward(self, x):
        # x is (Batch, 1, Channels, Samples)
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        # Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.depthwise(x)
        x = self.batchnorm2(x)
        x = self.elu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = self.sep_conv_depth(x)
        x = self.sep_conv_point(x)
        x = self.batchnorm3(x)
        x = self.elu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)

        # Transformer Injection
        # x shape: (batch, F2, 1, time)
        x = x.squeeze(2)         # (batch, F2, time)
        x = x.permute(0, 2, 1)   # (batch, time, F2)
        x = self.transformer(x)  # Self-Attention
        x = x.permute(0, 2, 1)   # (batch, F2, time)
        x = x.unsqueeze(2)       # (batch, F2, 1, time)

        # Classifier
        x = self.flatten(x)
        x = self.fc(x)
        return x

class EEGNet_Pipeline:
    def __init__(self, epochs=100, batch_size=32, lr=1e-3, channels=22, samples=656, num_classes=4,
                 device="auto", amp=True, patience=9999,
                 F1=32, D=2, F2=64, kernel_length=64, dropout=0.0, label_smoothing=0.0):
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = get_device(device)
        self.amp = amp and self.device.type == "cuda"
        self.patience = patience
        self.scaler = torch.amp.GradScaler("cuda") if self.amp else None
        self.model = EEGNet(
            num_classes=num_classes, channels=channels, samples=samples,
            F1=F1, D=D, F2=F2, kernel_length=kernel_length, p_drop=dropout
        ).to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.0)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(1, epochs), eta_min=1e-6)
        
        self.training_time = 0.0
        self.mean = None
        self.std = None
        self.history = {"loss": [], "val_loss": [], "acc": [], "val_acc": [], "lr": []}
        self.channels = channels

    def _prepare_data(self, X):
        X = np.asarray(X, dtype=np.float32)  # force float32 to halve memory
        if X.ndim == 3:
            X = X[:, np.newaxis, :, :]
        elif X.ndim == 2:
            X = X[np.newaxis, np.newaxis, :, :]
        return X

    def fit(self, X, y, X_val=None, y_val=None, progress_callback=None, incremental=False):
        X = self._prepare_data(X)
        
        n_out = self.model.num_classes
        y = np.asarray(y)
        
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
            
        class_counts = np.bincount(y, minlength=n_out)
        total_samples = len(y)
        class_weights = np.ones(len(class_counts), dtype=np.float32)
        for c in range(len(class_counts)):
            if class_counts[c] > 0:
                class_weights[c] = total_samples / (len(class_counts) * class_counts[c])
        self.criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(self.device))
        
        batch_mean = np.mean(X, axis=(0, 3), keepdims=True).astype(np.float32)
        batch_std = (np.std(X, axis=(0, 3), keepdims=True) + 1e-8).astype(np.float32)

        if incremental and self.mean is not None:
            alpha = 0.1
            m = np.asarray(self.mean, dtype=np.float32).reshape(1, -1, 1, 1)
            s = np.asarray(self.std, dtype=np.float32).reshape(1, -1, 1, 1)
            self.mean = (1 - alpha) * m + alpha * batch_mean
            self.std = (1 - alpha) * s + alpha * batch_std
        else:
            self.mean = batch_mean
            self.std = batch_std
            
        std = np.asarray(self.std, dtype=np.float32)
        std[std < 1e-6] = 1.0
        self.std = std

        X = (X - self.mean) / self.std
        
        dataset = torch.utils.data.TensorDataset(
            torch.tensor(X, dtype=torch.float32), 
            torch.tensor(y, dtype=torch.long)
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        if X_val is not None:
            X_val = self._prepare_data(X_val)
            X_val = (X_val - self.mean) / self.std
            val_X_t = torch.tensor(X_val, dtype=torch.float32).to(self.device)
            val_y_t = torch.tensor(y_val, dtype=torch.long).to(self.device)

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
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

            train_loss = running_loss / total
            train_acc = correct / total
            self.scheduler.step()

            val_loss, val_acc = 0.0, 0.0
            if X_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(val_X_t)
                    v_loss = self.criterion(val_outputs, val_y_t)
                    val_loss = v_loss.item()
                    _, v_pred = torch.max(val_outputs.data, 1)
                    val_acc = (v_pred == val_y_t).sum().item() / val_y_t.size(0)
                    
            self.history["loss"].append(train_loss)
            self.history["acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            try:
                self.history["lr"].append(self.optimizer.param_groups[0]["lr"])
            except Exception:
                pass
                
            print(json.dumps({
                "type": "epoch",
                "epoch": epoch + 1,
                "total_epochs": self.epochs,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc
            }), flush=True)
                
            # Early stopping effectively disabled (patience=9999 by default)
            if X_val is not None and self.patience < 9999:
                best = max(self.history["val_acc"]) if self.history["val_acc"] else val_acc
                bad = sum(1 for v in reversed(self.history["val_acc"]) if v < best)
                if bad >= self.patience:
                    if progress_callback:
                        progress_callback(epoch + 1, train_loss, train_acc, val_loss, val_acc)
                    break

            if progress_callback:
                progress_callback(epoch + 1, train_loss, train_acc, val_loss, val_acc)

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
            _, predicted = torch.max(outputs.data, 1)
            return predicted.cpu().numpy()

    def predict_proba(self, X):
        self.model.eval()
        X = self._prepare_data(X)
        if self.mean is not None:
            m = self.mean.cpu().numpy() if hasattr(self.mean, 'cpu') else self.mean
            s = self.std.cpu().numpy() if hasattr(self.std, 'cpu') else self.std
            X = (X - m) / s
        with torch.no_grad():
            if isinstance(X, torch.Tensor):
                X_t = X.clone().detach().to(torch.float32).to(self.device)
            else:
                X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(X_t)
            probs = torch.nn.functional.softmax(outputs, dim=1)
        return probs.cpu().numpy()
        
    def measure_latency(self, X_sample):
        self.model.eval()
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
            'channels': self.model.channels,
            'channel_names': getattr(self, 'channel_names', None),
            'sfreq': getattr(self, 'sfreq', 160.0),
            'num_classes': self.model.num_classes,
            'arch': 'eegnet',
            'model_cfg': {
                'F1': self.model.F1,
                'D': self.model.D,
                'F2': self.model.F2,
            },
        }
        torch.save(state, filepath)

    def load(self, filepath):
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        saved_cfg = state.get('model_cfg', {}) or {}
        
        channels = state.get('channels', 22)
        num_classes = state.get('num_classes', 4)

        self.model = EEGNet(
            num_classes=num_classes, channels=channels,
            F1=int(saved_cfg.get('F1', 8)),
            D=int(saved_cfg.get('D', 2)),
            F2=int(saved_cfg.get('F2', 16)),
        ).to(self.device)

        try:
            self.model.load_state_dict(state['model_state_dict'])
        except Exception as e:
            print("Warning: Could not load state_dict.", e)
        
        self.mean = state.get('mean', None)
        self.std = state.get('std', None)
        self.label_classes_ = state.get('label_classes', None)
        self.epochs = state.get('epochs', 100)
        self.batch_size = state.get('batch_size', 64)
        self.lr = state.get('lr', 1e-3)
        return self
