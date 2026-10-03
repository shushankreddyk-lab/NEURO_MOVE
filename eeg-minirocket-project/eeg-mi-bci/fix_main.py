import re
import os

app_path = r'D:\eeg-minirocket-project\eeg-mi-bci\dashboard\app.py'
with open(app_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Fix dataset_sel_tab1 in Training Process
text = text.replace('["Physionet", "BCI2a"]', '["High-Gamma Dataset", "Kaya Finger Movements", "WAY-EEG-GAL", "DREAMER Emotion"]')

# Fix models_dir assignment for new datasets
text = text.replace("models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', selected_dataset_tab1)", 
'''# Map dataset name to folder name
ds_folder_map = {
    "High-Gamma Dataset": "HighGamma",
    "Kaya Finger Movements": "Kaya",
    "WAY-EEG-GAL": "WAY",
    "DREAMER Emotion": "DREAMER"
}
mapped_folder = ds_folder_map.get(selected_dataset_tab1, selected_dataset_tab1)
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', mapped_folder)
''')

# In Live Training Console
text = text.replace('''        else:
            default_path = st.session_state.get('scanned_data_dir', r"D:\\eeg-minirocket-project\\physionet")
            folder_hint = "Enter root path to the 109-subject PhysioNet folder:"''', '''        else:
            default_path = st.session_state.get('scanned_data_dir', r"D:\\eeg-minirocket-project\\data")
            folder_hint = "Enter root path to the dataset folder:"''')

# Remove PhysioNet block in Tab 1
text = re.sub(r'        if "PhysioNet" in selected_dataset_str:.*?        elif "BCI" in selected_dataset_str:', '        if "BCI" in selected_dataset_str:', text, flags=re.DOTALL)
text = text.replace('        if "BCI" in selected_dataset_str:', '        if "High-Gamma" in selected_dataset_str:')

# Fix the c_icons and c_names in Tab 1
text = re.sub(r'    if "PhysioNet" in selected_dataset_str:.*?    elif "BCI" in selected_dataset_str:.*?    elif "High-Gamma"', '    if "High-Gamma"', text, flags=re.DOTALL)

with open(app_path, 'w', encoding='utf-8') as f:
    f.write(text)
print("Main app patched successfully")
