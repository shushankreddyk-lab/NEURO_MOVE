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

class ShallowConvNet(nn.Module):
    def __init__(self, channels=22, samples=656, num_classes=4, drop_prob=0.3):
        super(ShallowConvNet, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        
        # Temporal convolution
        self.conv_time = nn.Conv2d(1, 40, (1, 25), bias=False)
        # Spatial convolution
        self.conv_spat = nn.Conv2d(40, 40, (channels, 1), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(40)
        
        self.pool = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(drop_prob)
        
        # calculate dummy shape for fc
        dummy = torch.randn(1, 1, channels, samples)
        dummy = self.conv_time(dummy)
        dummy = self.conv_spat(dummy)
        dummy = self.batchnorm1(dummy)
        dummy = dummy ** 2
        dummy = self.pool(dummy)
        dummy = torch.log(torch.clamp(dummy, min=1e-6))
        dummy = self.dropout(dummy)
        
        out_dim = dummy.view(-1).shape[0]
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.batchnorm1(x)
        
        # Activation: square
        x = x ** 2
        
        x = self.pool(x)
        
        # Activation: log
        x = torch.log(torch.clamp(x, min=1e-6))
        
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class DeepConvNet(nn.Module):
    def __init__(self, channels=22, samples=656, num_classes=4, drop_prob=0.5):
        super(DeepConvNet, self).__init__()
        self.channels = channels
        self.num_classes = num_classes
        
        # Block 1
        self.conv_time = nn.Conv2d(1, 25, (1, 10), bias=False)
        self.conv_spat = nn.Conv2d(25, 25, (channels, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(25)
        self.pool1 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.drop1 = nn.Dropout(drop_prob)
        
        # Block 2
        self.conv2 = nn.Conv2d(25, 50, (1, 10), bias=False)
        self.bn2 = nn.BatchNorm2d(50)
        self.pool2 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.drop2 = nn.Dropout(drop_prob)
        
        # Block 3
        self.conv3 = nn.Conv2d(50, 100, (1, 10), bias=False)
        self.bn3 = nn.BatchNorm2d(100)
        self.pool3 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.drop3 = nn.Dropout(drop_prob)
        
        # Block 4
        self.conv4 = nn.Conv2d(100, 200, (1, 10), bias=False)
        self.bn4 = nn.BatchNorm2d(200)
        self.pool4 = nn.MaxPool2d((1, 3), stride=(1, 3))
        self.drop4 = nn.Dropout(drop_prob)
        
        # calculate dummy shape for fc
        dummy = torch.randn(1, 1, channels, samples)
        dummy = self.pool1(self.bn1(self.conv_spat(self.conv_time(dummy))))
        dummy = self.pool2(self.bn2(self.conv2(dummy)))
        dummy = self.pool3(self.bn3(self.conv3(dummy)))
        dummy = self.pool4(self.bn4(self.conv4(dummy)))
        
        out_dim = dummy.view(-1).shape[0]
        self.fc = nn.Linear(out_dim, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.conv_time(x)
        x = self.conv_spat(x)
        x = self.bn1(x)
        x = nn.functional.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = nn.functional.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = nn.functional.elu(x)
        x = self.pool3(x)
        x = self.drop3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = nn.functional.elu(x)
        x = self.pool4(x)
        x = self.drop4(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class ConvNet_Pipeline:
    def __init__(self, arch="shallow", epochs=100, batch_size=64, lr=1e-3, channels=22, samples=656, num_classes=4,
                 device="auto", amp=True, patience=9999):
        self.arch = arch
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = get_device(device)
        self.amp = amp and self.device.type == "cuda"
        self.patience = patience
        self.scaler = torch.amp.GradScaler("cuda") if self.amp else None
        
        if self.arch == "shallow":
            self.model = ShallowConvNet(channels=channels, samples=samples, num_classes=num_classes).to(self.device)
        else:
            self.model = DeepConvNet(channels=channels, samples=samples, num_classes=num_classes).to(self.device)
            
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
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
                y_val = np.array([lut.get(v, 0) for v in np.asarray(y_val)], dtype=np.int64)
            n_out = len(uniq)
        else:
            self.label_classes_ = np.arange(n_out)
            
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
            'num_classes': self.model.num_classes,
            'arch': self.arch,
        }
        torch.save(state, filepath)

    def load(self, filepath):
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        self.arch = state.get('arch', 'shallow')
        channels = state.get('channels', 22)
        num_classes = state.get('num_classes', 4)

        if self.arch == "shallow":
            self.model = ShallowConvNet(channels=channels, num_classes=num_classes).to(self.device)
        else:
            self.model = DeepConvNet(channels=channels, num_classes=num_classes).to(self.device)

        self.model.load_state_dict(state['model_state_dict'])
        self.mean = state.get('mean', None)
        self.std = state.get('std', None)
        self.label_classes_ = state.get('label_classes', None)
        return self
