import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGTransformer(nn.Module):
    """
    A Convolutional Transformer designed for EEG Motor Imagery.
    Extracts spatial/temporal features via a small CNN block, 
    then passes them to a Multi-Head Self-Attention Transformer Encoder.
    """
    def __init__(self, n_classes=4, in_chans=22, input_window_samples=1000, 
                 embed_dim=40, depth=2, heads=8, drop_rate=0.5):
        super().__init__()
        
        # 1. Convolutional Block (Spatial & Temporal Filtering)
        self.conv_temporal = nn.Conv2d(1, 40, (1, 25), padding=(0, 12))
        self.conv_spatial = nn.Conv2d(40, 40, (in_chans, 1), bias=False)
        self.batchnorm = nn.BatchNorm2d(40)
        self.pooling = nn.AvgPool2d((1, 75), stride=(1, 15))
        self.dropout = nn.Dropout(drop_rate)
        
        # Calculate sequence length after pooling
        out_len = ((input_window_samples - 75) // 15) + 1
        
        # 2. Transformer Encoder Block
        # We treat the flattened spatial filters as the embedding dimension
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, out_len + 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=drop_rate,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 3. Classification Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, n_classes)
        )

    def forward(self, x):
        # x shape: (Batch, Channels, Timepoints)
        # Add dummy dimension for Conv2d -> (Batch, 1, Channels, Timepoints)
        x = x.unsqueeze(1)
        
        # Convolutional feature extraction
        x = self.conv_temporal(x)
        x = self.conv_spatial(x)
        x = self.batchnorm(x)
        x = F.elu(x)
        x = self.pooling(x)
        x = self.dropout(x)
        
        # Reshape for Transformer: (Batch, SequenceLength, EmbeddingDim)
        x = x.squeeze(2).transpose(1, 2)
        
        # Add Class Token and Positional Embedding
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        
        # Pass through Transformer
        x = self.transformer_encoder(x)
        
        # Take the output of the CLS token for classification
        cls_out = x[:, 0, :]
        return self.mlp_head(cls_out)

if __name__ == "__main__":
    # Test the model with dummy data
    dummy_input = torch.randn(16, 22, 1000) # Batch=16, Chans=22, Time=1000
    model = EEGTransformer(n_classes=4, in_chans=22, input_window_samples=1000)
    out = model(dummy_input)
    print("Transformer Output Shape:", out.shape) # Should be (16, 4)

import time
import numpy as np
import os

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class CNN_Transformer_Pipeline:
    def __init__(self, num_classes=4, channels=20, samples=656, 
                 epochs=15, batch_size=64, lr=1e-3, device="cuda", task_type="classification"):
        self.device = get_device()
        self.num_classes = num_classes
        self.channels = channels
        self.samples = samples
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.task_type = task_type
        
        self.model = EEGTransformer(
            n_classes=num_classes, 
            in_chans=channels, 
            input_window_samples=samples
        ).to(self.device)
        
        self.classes_ = None
        self.training_time = 0.0
        
        self.feat_mean = None
        self.feat_std = None

    def fit(self, X, y, X_val=None, y_val=None):
        start_time = time.time()
        
        self.classes_ = np.unique(y) if self.task_type == "classification" else np.arange(y.shape[1] if y.ndim > 1 else 1)
        
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
        dl = torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        
        if X_val is not None and y_val is not None:
            X_v = torch.tensor(X_val, dtype=torch.float32)
            X_v = (X_v - self.feat_mean) / self.feat_std
            if self.task_type == "classification":
                y_v = torch.tensor([lut[v] for v in y_val], dtype=y_dtype)
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
                import json
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
            'samples': self.samples,
            'task_type': getattr(self, 'task_type', 'classification')
        }
        torch.save(state, filepath)

    def load(self, filepath):
        state = torch.load(filepath, map_location=self.device, weights_only=False)
        self.num_classes = state.get('num_classes', self.num_classes)
        self.channels = state.get('channels', self.channels)
        self.samples = state.get('samples', self.samples)
        self.task_type = state.get('task_type', self.task_type)
        self.feat_mean = state.get('feat_mean', None)
        self.feat_std = state.get('feat_std', None)
        
        self.model = EEGTransformer(self.num_classes, self.channels, self.samples).to(self.device)
        sd_key = 'model' if 'model' in state else 'model_state_dict' if 'model_state_dict' in state else None
        self.model.load_state_dict(state[sd_key])

        if self.feat_mean is None: self.feat_mean = state.get('feat_mean')
        if self.feat_std is None: self.feat_std = state.get('feat_std')
        self.classes_ = state.get('classes_')
        self.training_time = state.get('training_time', 0.0)
        
        self.model.eval()
        return self

