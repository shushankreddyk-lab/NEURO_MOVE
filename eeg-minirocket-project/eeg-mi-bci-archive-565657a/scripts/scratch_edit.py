import os
import json

filepath = r'd:\eeg-minirocket-project\eeg-mi-bci-archive-565657a\src\train_master.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

new_callback = '''    def progress_callback(epoch, train_loss, train_acc, val_loss, val_acc):
        print(json.dumps({
            "type": "epoch",
            "epoch": f"{epoch}/{epochs}",
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc)
        }), flush=True)

    # --- MODEL EXECUTION SWITCH ---'''
content = content.replace('    # --- MODEL EXECUTION SWITCH ---', new_callback)
content = content.replace('cnn_lstm_pipeline.fit(X_train, y_train, X_val, y_val)', 'cnn_lstm_pipeline.fit(X_train, y_train, X_val, y_val, progress_callback=progress_callback)')
content = content.replace('conformer_pipeline.fit(X_train, y_train, X_val, y_val)', 'conformer_pipeline.fit(X_train, y_train, X_val, y_val, progress_callback=progress_callback)')
content = content.replace('eegnet_pipeline.fit(X_train, y_train, X_val, y_val)', 'eegnet_pipeline.fit(X_train, y_train, X_val, y_val, progress_callback=progress_callback)')
content = content.replace('shallow_pipeline.fit(X_train, y_train, X_val, y_val)', 'shallow_pipeline.fit(X_train, y_train, X_val, y_val, progress_callback=progress_callback)')

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print('Done')
