import numpy as np
import sys
sys.path.append('src')
from cnn_lstm_engine import CNN_LSTM_Pipeline

# Same splitting logic as train_master.py
data = np.load(r'D:\eeg-minirocket-project\eeg-mi-bci-archive-565657a\data\processed\data_physionetmi_subs1to1_79939687.npz', mmap_mode='r', allow_pickle=True)
X = np.array(data['X'], dtype=np.float32)
y = np.array(data['y'])
meta = data.get('meta')

X_train, y_train, X_val, y_val, X_test, y_test = [], [], [], [], [], []
for subject_id in np.unique([m['subject_id'] for m in meta]):
    sub_mask = np.array([m['subject_id'] == subject_id for m in meta])
    sub_X = X[sub_mask]
    sub_y = y[sub_mask]
    sub_meta = meta[sub_mask]
    sub_runs = np.sort(np.unique([m['run_id'] for m in sub_meta]))
    test_runs = [sub_runs[-1]]
    val_runs = [sub_runs[-2]]
    train_runs = sub_runs[:-2]
    for i, m in enumerate(sub_meta):
        if m['run_id'] in test_runs:
            X_test.append(sub_X[i]); y_test.append(sub_y[i])
        elif m['run_id'] in val_runs:
            X_val.append(sub_X[i]); y_val.append(sub_y[i])
        else:
            X_train.append(sub_X[i]); y_train.append(sub_y[i])

X_train = np.array(X_train)
y_train = np.array(y_train)

# Channel selection as in train_master.py
variances = np.var(X_train, axis=(0, 2))
best_channels_idx = np.argsort(variances)[-20:]
best_channels_idx = np.sort(best_channels_idx)
X_train = X_train[:, best_channels_idx, :]
X_test = np.array(X_test)[:, best_channels_idx, :]
y_test = np.array(y_test)

print(f"X_train[0] mean: {np.mean(X_train[0])}, std: {np.std(X_train[0])}")

model = CNN_LSTM_Pipeline(device='cpu').load(r'models\master_physionet_cnn_lstm_subs1to1_20261004_191111.pth')
pred = model.predict(X_train[:10])
print(f"y_train[:10]  : {y_train[:10]}")
print(f"pred_train[:10]: {pred}")

pred_test = model.predict(X_test)
print(f"y_test  : {y_test}")
print(f"pred_test: {pred_test}")
