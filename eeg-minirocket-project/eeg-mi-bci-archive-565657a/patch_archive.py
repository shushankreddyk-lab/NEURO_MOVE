import os

app_path = r'D:\eeg-minirocket-project\eeg-mi-bci-archive-565657a\dashboard\app.py'
with open(app_path, 'r', encoding='utf-8') as f:
    text = f.read()

# 1. Update options array
old_opts = "options=['🧠 Overview','🏗️ Architectures','💻 Dataset Console','🚀 Live Training'"
new_opts = "options=['🧠 Overview','🏗️ Architectures','💻 Dataset Console','📊 Training Process','🚀 Live Training'"
text = text.replace(old_opts, new_opts)

# 2. Update icons array
old_icons = "icons=['brain','layers','terminal','rocket'"
new_icons = "icons=['brain','layers','terminal','graph-up','rocket'"
text = text.replace(old_icons, new_icons)

# 3. Inject snippet before Live Training tab
with open(r'D:\eeg-minirocket-project\eeg-mi-bci-archive-565657a\training_process_snippet.py', 'r', encoding='utf-8') as f:
    snippet = f.read()

# Replace selected_dataset_tab1 with 'Physionet' and 'BCI2a' options
snippet = snippet.replace('["High-Gamma Dataset", "Kaya Finger Movements", "WAY-EEG-GAL"]', '["Physionet", "BCI2a"]')

# Remove the ds_folder_map code if it exists (since we patched it in the main app, but we extracted it from the patched main app)
snippet = snippet.replace('''# Map dataset name to folder name
ds_folder_map = {
    "High-Gamma Dataset": "HighGamma",
    "Kaya Finger Movements": "Kaya",
    "WAY-EEG-GAL": "WAY",
    "": ""
}
mapped_folder = ds_folder_map.get(selected_dataset_tab1, selected_dataset_tab1)
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', mapped_folder)''', 
"models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', selected_dataset_tab1)")

# Convert `if selected_tab == '📊 Training Process':` to `elif`
snippet = snippet.replace("if selected_tab == '📊 Training Process':", "elif selected_tab == '📊 Training Process':")

# Insert before `elif selected_tab == '🚀 Live Training':`
inject_target = "if selected_tab == '🚀 Live Training':"
if inject_target in text:
    text = text.replace(inject_target, snippet + '\n' + "elif selected_tab == '🚀 Live Training':")
else:
    print("Could not find insertion target!")

with open(app_path, 'w', encoding='utf-8') as f:
    f.write(text)
print("Archive app patched successfully")
