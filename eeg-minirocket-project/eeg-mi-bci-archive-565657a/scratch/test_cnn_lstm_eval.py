import numpy as np
import torch
import sys
sys.path.append('src')
from cnn_lstm_engine import CNN_LSTM_Pipeline

data = np.load(r'D:\eeg-minirocket-project\eeg-mi-bci-archive-565657a\data\processed\data_physionetmi_subs1to1_79939687.npz')
X = data['X'][:10]
y = data['y'][:10]

model = CNN_LSTM_Pipeline(device='cpu').load(r'models\master_physionet_cnn_lstm_subs1to1_20261004_191111.pth')
print("Ground truth:", y)

# Predict normally (which uses model.eval())
print("Predict proba (eval mode):")
probs = model.predict_proba(X)
print(np.argmax(probs, axis=1))

# Predict with train mode!
model.model.train()
with torch.no_grad():
    X_prep = model._prepare_data(X)
    X_prep = (X_prep - model.mean) / model.std
    X_t = torch.tensor(X_prep, dtype=torch.float32)
    out = model.model(X_t)
    print("Predict proba (train mode):")
    print(torch.argmax(out, dim=1).numpy())
