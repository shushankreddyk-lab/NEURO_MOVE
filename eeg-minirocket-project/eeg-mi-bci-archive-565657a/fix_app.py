import os
import re

filepath = r'd:\eeg-minirocket-project\eeg-mi-bci-archive-565657a\dashboard\app.py'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Fix the file upload extension bug in the first uploader
content = content.replace("ext = '.gdf' if uploaded_file.name.lower().endswith('.gdf') else '.edf'", "ext = os.path.splitext(uploaded_file.name)[1].lower()")

# 2. Fix the file upload for 3D Topology Analysis
content = content.replace('st.file_uploader("Upload EDF/GDF for 3D Topology Analysis", type=["edf", "gdf"]', 'st.file_uploader("Upload EEG file for 3D Topology Analysis", type=["edf", "gdf", "mat", "csv"]')

content = content.replace("ext = '.gdf' if uploaded_edf.name.lower().endswith('.gdf') else '.edf'", "ext = os.path.splitext(uploaded_edf.name)[1].lower()")

# 3. Add the missing models to the Select Model dropdown
content = content.replace('["MiniRocket Pipeline", "EEG-Conformer", "13-Layer CNN-LSTM"]', '["MiniRocket Pipeline", "EEG-Conformer", "13-Layer CNN-LSTM", "EEGNet", "Shallow ConvNet"]')

# We also need to fix how it loads mat/csv in Tab 6, because it doesn't currently support them
# Wait, let's see how tab 6 handles it. I'll just leave it or insert the same parsing logic.
tab6_old_logic = """if ext == '.gdf':
    raw = mne.io.read_raw_gdf(temp_path, preload=True, verbose=False)
else:
    raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)"""

tab6_new_logic = """import scipy.io
import pandas as pd
if ext == '.gdf':
    raw = mne.io.read_raw_gdf(temp_path, preload=True, verbose=False)
elif ext == '.edf':
    raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)
elif ext == '.csv':
    df = pd.read_csv(temp_path)
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
    data = df.values.T if df.shape[1] < df.shape[0] else df.values
    info = mne.create_info(ch_names=[str(i) for i in range(data.shape[0])], sfreq=250.0, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
elif ext == '.mat':
    mat = scipy.io.loadmat(temp_path)
    if 'o' in mat:
        data = mat['o'][0,0]['data'].T
        info = mne.create_info(ch_names=[str(i) for i in range(data.shape[0])], sfreq=1000.0, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
    elif 'DREAMER' in mat:
        data = mat['DREAMER'][0,0]['Data'][0,0]['EEG'][0,0]['stimuli'][0,0].T
        sfreq = int(mat['DREAMER'][0,0]['EEG_SamplingRate'][0,0])
        info = mne.create_info(ch_names=[str(i) for i in range(data.shape[0])], sfreq=sfreq, ch_types='eeg')
        raw = mne.io.RawArray(data, info)
    else:
        raise ValueError("Unknown .mat format")
else:
    raise ValueError(f"Unsupported ext: {ext}")"""
    
content = content.replace(tab6_old_logic, tab6_new_logic)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print('Done app.py edits')
