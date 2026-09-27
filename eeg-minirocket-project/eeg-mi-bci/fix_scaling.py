import re

# Fix train_master.py
file_path = r'd:\eeg-minirocket-project\eeg-mi-bci\src\train_master.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace X_batch = epochs.get_data(copy=False) * 1e6 with X_batch = epochs.get_data(copy=False)
content = content.replace("X_batch = epochs.get_data(copy=False) * 1e6", "X_batch = epochs.get_data(copy=False)")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

# Fix app.py
file_path = r'd:\eeg-minirocket-project\eeg-mi-bci\dashboard\app.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace X = epochs.get_data(copy=False) * 1e6 with X = epochs.get_data(copy=False)
content = content.replace("X = epochs.get_data(copy=False) * 1e6", "X = epochs.get_data(copy=False)")

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Removed double scaling.")
